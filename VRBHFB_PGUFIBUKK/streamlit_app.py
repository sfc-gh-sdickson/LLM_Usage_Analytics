# Import python packages
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, sum as spark_sum, count
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import json
from typing import Dict, List, Tuple

#############################################
#     CONFIGURATION & SETUP
#############################################
st.set_page_config(
    page_title="Snowflake LLM Usage Analytics",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .stMetric > div {
        font-size: 24px !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session
@st.cache_resource
def get_session():
    return get_active_session()

session = get_session()

#############################################
#     HELPER FUNCTIONS
#############################################
@st.cache_data(ttl=300)  # Cache for 5 minutes

def read_svg(path):
            """Helper function to read SVG files"""
            with open(path, 'r') as f:
                svg_string = f.read()
            return svg_string

def execute_query(query: str) -> pd.DataFrame:
    """Execute query with caching and error handling"""
    try:
        result = session.sql(query)
        return result.to_pandas()
    except Exception as e:
        st.error(f"Query execution failed: {str(e)}")
        return pd.DataFrame()

def format_number(num) -> str:
    """Format numbers with appropriate suffixes"""
    if pd.isna(num) or num == 0:
        return "0"

    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,.0f}"

def create_date_filter() -> Tuple[datetime.date, datetime.date]:
    """Create enhanced date filter with presets"""
    max_date = datetime.datetime.now().date()
    min_date = max_date - datetime.timedelta(days=365)

    # Initialize session state
    if 'date_range' not in st.session_state:
        st.session_state.date_range = (
            max_date - datetime.timedelta(days=30),
            max_date
        )

    with st.sidebar:
        svg_content2 = read_svg("Snowflake_Logo.svg")
        st.image(svg_content2, width=150)
        st.header("📅 Date Range Filter")

        # Quick preset buttons
        st.subheader("Quick Presets")
        col1, col2 = st.columns(2)

        presets = {
            "7D": 7, "30D": 30, "60D": 60, "90D": 90,
            "6M": 180, "1Y": 365
        }

        for i, (label, days) in enumerate(presets.items()):
            col = col1 if i % 2 == 0 else col2
            if col.button(label, key=f"preset_{label}"):
                st.session_state.date_range = (
                    max_date - datetime.timedelta(days=days),
                    max_date
                )
                st.rerun()

        # Custom date picker
        st.subheader("Custom Range")
        date_range = st.date_input(
            "Select date range:",
            value=st.session_state.date_range,
            min_value=min_date,
            max_value=max_date,
            key="date_picker"
        )

        if len(date_range) == 2:
            st.session_state.date_range = date_range
            return date_range
        else:
            return st.session_state.date_range


#############################################
#     MAIN APP
#############################################
def main():
    # Header
    svg_content = read_svg("Snowflake.svg")
    st.image(svg_content, width=75)
    st.title("Snowflake LLM Usage Analytics")
    st.markdown("**Comprehensive analysis of Cortex AI Services usage and costs**")

    # Date filter
    start_date, end_date = create_date_filter()

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Additional Filters")

        # Get available warehouses for filtering
        wh_query = f"""
        SELECT DISTINCT WAREHOUSE_NAME 
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY 
        WHERE START_TIME BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY WAREHOUSE_NAME
        """
        warehouses_df = execute_query(wh_query)

        if not warehouses_df.empty:
            selected_warehouses = st.multiselect(
                "Filter by Warehouse:",
                options=warehouses_df['WAREHOUSE_NAME'].tolist(),
                default=warehouses_df['WAREHOUSE_NAME'].tolist()[:10],  # Limit default selection
                key="warehouse_filter"
            )
        else:
            selected_warehouses = []

    # Main content
    display_overview_metrics(start_date, end_date)
    display_usage_analytics(start_date, end_date, selected_warehouses)
    display_detailed_analysis(start_date, end_date)
    display_cortex_services(start_date, end_date)

def display_overview_metrics(start_date: datetime.date, end_date: datetime.date):
    """Display high-level overview metrics"""
    st.header("📊 Overview Metrics")

    # Query for overview metrics
    overview_query = f"""
    WITH ai_services AS (
        SELECT COALESCE(SUM(credits_used), 0) as ai_credits
        FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY 
        WHERE start_time BETWEEN '{start_date}' AND '{end_date}' 
        AND SERVICE_TYPE = 'AI_SERVICES'
    ),
    cortex_complete AS (
        SELECT 
            COALESCE(SUM(token_credits), 0) as complete_credits,
            COALESCE(SUM(tokens), 0) as total_tokens,
            COUNT(*) as total_requests
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
        WHERE function_name = 'COMPLETE' 
        AND start_time BETWEEN '{start_date}' AND '{end_date}'
    ),
    cortex_analyst AS (
        SELECT 
            COALESCE(SUM(credits), 0) as analyst_credits,
            COALESCE(SUM(request_count), 0) as analyst_requests
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY 
        WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    ),
    cortex_search AS (
        SELECT COALESCE(SUM(credits), 0) as search_credits
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY 
        WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    )
    SELECT 
        ai.ai_credits,
        cc.complete_credits,
        cc.total_tokens,
        cc.total_requests,
        ca.analyst_credits,
        ca.analyst_requests,
        cs.search_credits,
        (cc.complete_credits + ca.analyst_credits + cs.search_credits) as total_cortex_credits
    FROM ai_services ai
    CROSS JOIN cortex_complete cc
    CROSS JOIN cortex_analyst ca
    CROSS JOIN cortex_search cs
    """

    metrics_df = execute_query(overview_query)

    if not metrics_df.empty:
        row = metrics_df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🔥 Total AI Services Credits",
                format_number(row['AI_CREDITS']),
                help="Total credits consumed by all AI services"
            )

        with col2:
            st.metric(
                "🤖 Cortex Complete Credits",
                format_number(row['COMPLETE_CREDITS']),
                help="Credits used for LLM inference"
            )

        with col3:
            st.metric(
                "📝 Total Tokens Processed",
                format_number(row['TOTAL_TOKENS']),
                help="Total tokens processed by Cortex Complete"
            )

        with col4:
            st.metric(
                "📊 Total LLM Requests",
                format_number(row['TOTAL_REQUESTS']),
                help="Total number of LLM completion requests"
            )

        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "🔍 Cortex Analyst Credits",
                format_number(row['ANALYST_CREDITS']),
                help="Credits used for Cortex Analyst"
            )

        with col6:
            st.metric(
                "🔎 Cortex Search Credits",
                format_number(row['SEARCH_CREDITS']),
                help="Credits used for Cortex Search"
            )

        with col7:
            avg_tokens_per_request = row['TOTAL_TOKENS'] / max(row['TOTAL_REQUESTS'], 1)
            st.metric(
                "📏 Avg Tokens/Request",
                format_number(avg_tokens_per_request),
                help="Average tokens per completion request"
            )

        with col8:
            cost_per_token = row['COMPLETE_CREDITS'] / max(row['TOTAL_TOKENS'], 1)
            st.metric(
                "💰 Credits/Token",
                f"{cost_per_token:.6f}",
                help="Average credits per token"
            )

def display_usage_analytics(start_date: datetime.date, end_date: datetime.date, selected_warehouses: List[str]):
    """Display usage analytics with improved visualizations"""
    st.header("📈 Usage Analytics")

    # Model usage analysis
    model_usage_query = f"""
    SELECT 
        model_name,
        SUM(token_credits) as total_credits,
        SUM(tokens) as total_tokens,
        COUNT(*) as request_count,
        AVG(tokens) as avg_tokens_per_request
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE function_name = 'COMPLETE' 
    AND start_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY model_name 
    ORDER BY total_credits DESC
    """

    model_df = execute_query(model_usage_query)

    if not model_df.empty:
        # Create subplots for model analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Credits by Model', 'Tokens by Model', 
                          'Requests by Model', 'Avg Tokens per Request'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Credits by model
        fig.add_trace(
            go.Bar(x=model_df['MODEL_NAME'], y=model_df['TOTAL_CREDITS'], 
                   name='Credits', marker_color='#1f77b4'),
            row=1, col=1
        )

        # Tokens by model
        fig.add_trace(
            go.Bar(x=model_df['MODEL_NAME'], y=model_df['TOTAL_TOKENS'], 
                   name='Tokens', marker_color='#ff7f0e'),
            row=1, col=2
        )

        # Requests by model
        fig.add_trace(
            go.Bar(x=model_df['MODEL_NAME'], y=model_df['REQUEST_COUNT'], 
                   name='Requests', marker_color='#2ca02c'),
            row=2, col=1
        )

        # Avg tokens per request
        fig.add_trace(
            go.Bar(x=model_df['MODEL_NAME'], y=model_df['AVG_TOKENS_PER_REQUEST'], 
                   name='Avg Tokens', marker_color='#d62728'),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=False, title_text="Model Usage Analysis")
        st.plotly_chart(fig, use_container_width=True)

    # Time series analysis
    display_time_series_analysis(start_date, end_date)

    # Warehouse analysis
    if selected_warehouses:
        display_warehouse_analysis(start_date, end_date, selected_warehouses)

def display_time_series_analysis(start_date: datetime.date, end_date: datetime.date):
    """Display time series analysis of usage patterns"""
    st.subheader("📅 Usage Trends Over Time")

    time_series_query = f"""
    SELECT 
        DATE(start_time) as usage_date,
        SUM(token_credits) as daily_credits,
        SUM(tokens) as daily_tokens,
        COUNT(*) as daily_requests
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE function_name = 'COMPLETE' 
    AND start_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY DATE(start_time)
    ORDER BY usage_date
    """

    ts_df = execute_query(time_series_query)

    if not ts_df.empty:
        # Create time series chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Daily Credits', 'Daily Tokens', 'Daily Requests'),
            vertical_spacing=0.08
        )

        fig.add_trace(
            go.Scatter(x=ts_df['USAGE_DATE'], y=ts_df['DAILY_CREDITS'], 
                      mode='lines+markers', name='Credits', line=dict(color='#1f77b4')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=ts_df['USAGE_DATE'], y=ts_df['DAILY_TOKENS'], 
                      mode='lines+markers', name='Tokens', line=dict(color='#ff7f0e')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=ts_df['USAGE_DATE'], y=ts_df['DAILY_REQUESTS'], 
                      mode='lines+markers', name='Requests', line=dict(color='#2ca02c')),
            row=3, col=1
        )

        fig.update_layout(height=600, showlegend=False, title_text="Usage Trends")
        fig.update_xaxes(title_text="Date", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

def display_warehouse_analysis(start_date: datetime.date, end_date: datetime.date, selected_warehouses: List[str]):
    """Display warehouse-specific analysis - FIXED VERSION"""
    st.subheader("🏭 Warehouse Analysis")

    warehouse_filter = "'" + "','".join(selected_warehouses) + "'"

    # Fixed query - using warehouse_id to join instead of query_id
    warehouse_query = f"""
    WITH compute_credits AS (
        SELECT 
            warehouse_name,
            warehouse_id,
            SUM(credits_used) as compute_credits
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY 
        WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
        AND warehouse_name IN ({warehouse_filter})
        GROUP BY warehouse_name, warehouse_id
    ),
    cortex_credits AS (
        SELECT 
            warehouse_id,
            SUM(token_credits) as cortex_credits,
            SUM(tokens) as cortex_tokens,
            COUNT(*) as cortex_requests
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
        WHERE function_name = 'COMPLETE' 
        AND start_time BETWEEN '{start_date}' AND '{end_date}'
        AND warehouse_id IS NOT NULL
        GROUP BY warehouse_id
    )
    SELECT 
        cc.warehouse_name,
        COALESCE(cc.compute_credits, 0) as compute_credits,
        COALESCE(ctx.cortex_credits, 0) as cortex_credits,
        COALESCE(ctx.cortex_tokens, 0) as cortex_tokens,
        COALESCE(ctx.cortex_requests, 0) as cortex_requests,
        (COALESCE(cc.compute_credits, 0) + COALESCE(ctx.cortex_credits, 0)) as total_credits
    FROM compute_credits cc
    LEFT JOIN cortex_credits ctx ON cc.warehouse_id = ctx.warehouse_id
    ORDER BY total_credits DESC
    """

    wh_df = execute_query(warehouse_query)

    if not wh_df.empty:
        # Warehouse comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Compute Credits',
            x=wh_df['WAREHOUSE_NAME'],
            y=wh_df['COMPUTE_CREDITS'],
            marker_color='#1f77b4'
        ))

        fig.add_trace(go.Bar(
            name='Cortex Credits',
            x=wh_df['WAREHOUSE_NAME'],
            y=wh_df['CORTEX_CREDITS'],
            marker_color='#ff7f0e'
        ))

        fig.update_layout(
            title='Credits by Warehouse (Compute vs Cortex)',
            barmode='stack',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Warehouse details table
        st.dataframe(
            wh_df.style.format({
                'COMPUTE_CREDITS': '{:.2f}',
                'CORTEX_CREDITS': '{:.2f}',
                'TOTAL_CREDITS': '{:.2f}',
                'CORTEX_TOKENS': '{:,.0f}',
                'CORTEX_REQUESTS': '{:,.0f}'
            }),
            use_container_width=True
        )

def display_detailed_analysis(start_date: datetime.date, end_date: datetime.date):
    """Display detailed analysis - FIXED VERSION"""
    st.header("🔍 Detailed Analysis")

    # Warehouse-based user analysis (since we can't join on query_id)
    warehouse_usage_query = f"""
    WITH warehouse_cortex AS (
        SELECT 
            warehouse_id,
            SUM(token_credits) as total_credits,
            SUM(tokens) as total_tokens,
            COUNT(*) as total_requests,
            model_name
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
        WHERE function_name = 'COMPLETE' 
        AND start_time BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY warehouse_id, model_name
    ),
    warehouse_info AS (
        SELECT DISTINCT 
            warehouse_id,
            warehouse_name
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
        WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    )
    SELECT 
        wi.warehouse_name,
        wc.model_name,
        wc.total_credits,
        wc.total_tokens,
        wc.total_requests,
        (wc.total_tokens / NULLIF(wc.total_requests, 0)) as avg_tokens_per_request
    FROM warehouse_cortex wc
    JOIN warehouse_info wi ON wc.warehouse_id = wi.warehouse_id
    ORDER BY wc.total_credits DESC
    LIMIT 50
    """

    usage_df = execute_query(warehouse_usage_query)

    if not usage_df.empty:
        st.subheader("🏭 Usage by Warehouse and Model")

        # Model distribution
        model_summary = usage_df.groupby('MODEL_NAME').agg({
            'TOTAL_CREDITS': 'sum',
            'TOTAL_TOKENS': 'sum',
            'TOTAL_REQUESTS': 'sum'
        }).reset_index()

        if not model_summary.empty:
            fig = px.pie(
                model_summary, 
                values='TOTAL_CREDITS', 
                names='MODEL_NAME',
                title='Credit Distribution by Model'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed usage table
        st.dataframe(
            usage_df.style.format({
                'TOTAL_CREDITS': '{:.2f}',
                'TOTAL_TOKENS': '{:,.0f}',
                'TOTAL_REQUESTS': '{:,.0f}',
                'AVG_TOKENS_PER_REQUEST': '{:.0f}'
            }),
            use_container_width=True
        )

    # Model performance analysis
    st.subheader("🤖 Model Performance Analysis")

    model_perf_query = f"""
    SELECT 
        model_name,
        DATE(start_time) as usage_date,
        SUM(token_credits) as daily_credits,
        SUM(tokens) as daily_tokens,
        COUNT(*) as daily_requests,
        AVG(tokens) as avg_tokens_per_request,
        (SUM(token_credits) / NULLIF(SUM(tokens), 0)) as credits_per_token
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE function_name = 'COMPLETE' 
    AND start_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY model_name, DATE(start_time)
    ORDER BY usage_date DESC, daily_credits DESC
    """

    perf_df = execute_query(model_perf_query)

    if not perf_df.empty:
        # Model selection for detailed view
        selected_models = st.multiselect(
            "Select Models for Detailed Analysis:",
            options=perf_df['MODEL_NAME'].unique().tolist(),
            default=perf_df['MODEL_NAME'].unique().tolist()[:3],
            key="model_detail_filter"
        )

        if selected_models:
            filtered_perf_df = perf_df[perf_df['MODEL_NAME'].isin(selected_models)]

            # Time series by model
            fig = px.line(
                filtered_perf_df, 
                x='USAGE_DATE', 
                y='DAILY_CREDITS',
                color='MODEL_NAME',
                title='Daily Credits by Model Over Time',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Performance metrics table
            st.dataframe(
                filtered_perf_df.style.format({
                    'DAILY_CREDITS': '{:.2f}',
                    'DAILY_TOKENS': '{:,.0f}',
                    'DAILY_REQUESTS': '{:,.0f}',
                    'AVG_TOKENS_PER_REQUEST': '{:.0f}',
                    'CREDITS_PER_TOKEN': '{:.6f}'
                }),
                use_container_width=True
            )

def display_cortex_services(start_date: datetime.date, end_date: datetime.date):
    """Display Cortex Analyst and Search analytics"""
    st.header("🧠 Cortex Services Analytics")

    # Cortex Analyst
    st.subheader("📊 Cortex Analyst")

    analyst_query = f"""
    SELECT 
        DATE(start_time) as usage_date,
        SUM(credits) as daily_credits,
        SUM(request_count) as daily_requests
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY DATE(start_time)
    ORDER BY usage_date
    """

    analyst_df = execute_query(analyst_query)

    if not analyst_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            total_analyst_credits = analyst_df['DAILY_CREDITS'].sum()
            st.metric("Total Analyst Credits", format_number(total_analyst_credits))

        with col2:
            total_analyst_requests = analyst_df['DAILY_REQUESTS'].sum()
            st.metric("Total Analyst Requests", format_number(total_analyst_requests))

        # Analyst usage chart
        fig = px.line(
            analyst_df, 
            x='USAGE_DATE', 
            y='DAILY_REQUESTS',
            title='Cortex Analyst Daily Usage',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Cortex Analyst usage data found for the selected date range.")

    # Cortex Search
    st.subheader("🔎 Cortex Search")

    search_query = f"""
    SELECT 
        service_name,
        SUM(credits) as total_credits,
        DATE(start_time) as usage_date
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY service_name, DATE(start_time)
    ORDER BY total_credits DESC
    """

    search_df = execute_query(search_query)

    if not search_df.empty:
        total_search_credits = search_df['TOTAL_CREDITS'].sum()
        st.metric("Total Search Credits", format_number(total_search_credits))

        # Search usage by service
        service_summary = search_df.groupby('SERVICE_NAME')['TOTAL_CREDITS'].sum().reset_index()

        if len(service_summary) > 0:
            fig = px.pie(
                service_summary, 
                values='TOTAL_CREDITS', 
                names='SERVICE_NAME',
                title='Search Credits by Service'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Daily search usage
        daily_search = search_df.groupby('USAGE_DATE')['TOTAL_CREDITS'].sum().reset_index()
        if not daily_search.empty:
            fig = px.line(
                daily_search,
                x='USAGE_DATE',
                y='TOTAL_CREDITS',
                title='Daily Search Credits Usage',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Cortex Search usage data found for the selected date range.")

    # Cost optimization recommendations
    display_cost_optimization_recommendations(start_date, end_date)

def display_cost_optimization_recommendations(start_date: datetime.date, end_date: datetime.date):
    """Display cost optimization recommendations"""
    st.subheader("💡 Cost Optimization Recommendations")

    # Get model efficiency data
    efficiency_query = f"""
    SELECT 
        model_name,
        SUM(token_credits) as total_credits,
        SUM(tokens) as total_tokens,
        COUNT(*) as total_requests,
        (SUM(token_credits) / NULLIF(SUM(tokens), 0)) as credits_per_token,
        AVG(tokens) as avg_tokens_per_request
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE function_name = 'COMPLETE' 
    AND start_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY model_name
    ORDER BY total_credits DESC
    """

    efficiency_df = execute_query(efficiency_query)

    if not efficiency_df.empty:
        recommendations = []

        # Find most expensive model per token
        most_expensive = efficiency_df.loc[efficiency_df['CREDITS_PER_TOKEN'].idxmax()]
        least_expensive = efficiency_df.loc[efficiency_df['CREDITS_PER_TOKEN'].idxmin()]

        recommendations.append({
            "type": "💰 Cost Efficiency",
            "recommendation": f"Consider switching from {most_expensive['MODEL_NAME']} to {least_expensive['MODEL_NAME']} for cost-sensitive workloads",
            "impact": f"Potential savings: {((most_expensive['CREDITS_PER_TOKEN'] - least_expensive['CREDITS_PER_TOKEN']) / most_expensive['CREDITS_PER_TOKEN'] * 100):.1f}% per token"
        })

        # Find models with high token usage
        high_usage_models = efficiency_df[efficiency_df['TOTAL_TOKENS'] > efficiency_df['TOTAL_TOKENS'].quantile(0.8)]
        if not high_usage_models.empty:
            recommendations.append({
                "type": "📊 Usage Optimization",
                "recommendation": f"High usage detected for: {', '.join(high_usage_models['MODEL_NAME'].tolist())}",
                "impact": "Consider implementing caching or batch processing for these models"
            })

        # Display recommendations
        for rec in recommendations:
            with st.expander(f"{rec['type']}: {rec['recommendation'][:50]}..."):
                st.write(f"**Recommendation:** {rec['recommendation']}")
                st.write(f"**Potential Impact:** {rec['impact']}")

# Footer
def display_footer():
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Version:** 2.1 Fixed")
    with col2:
        st.markdown("**Updated:** March 2025")
    with col3:
        st.markdown("**Data Source:** Snowflake Account Usage")

    st.markdown("---")
    st.markdown("**Note:** This dashboard analyzes Cortex AI Services usage. Ensure you have appropriate permissions to access ACCOUNT_USAGE views.")

if __name__ == "__main__":
    main()
    display_footer()