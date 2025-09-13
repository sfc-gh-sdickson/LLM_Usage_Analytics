<img src="Snowflake_Logo.svg" width="200">
# Snowflake LLM Usage Analytics Dashboard

## ❄️ Updates

*   **Snowflake Github Updates:** I added my standard Snowflake logos in the Snowflake Streamlit editor and then pushed it to Github.  Snowflake create a directory for the new files: VRBHFB_PGUFIBUKK

This Streamlit application provides a comprehensive dashboard to monitor and analyze the usage and costs associated with Snowflake's Cortex AI Services. It offers detailed visualizations and insights into your organization's LLM consumption, helping you optimize costs and understand usage patterns.

## ❄️ Features

*   **Overview Metrics:** Get a high-level view of total credits consumed, tokens processed, and the number of LLM requests.
*   **Usage Analytics:** Dive deep into usage patterns with breakdowns by:
    *   **Model:** Analyze credits, tokens, and requests per LLM model.
    *   **Time:** Visualize usage trends over time with daily granularity.
    *   **Warehouse:** Compare compute and Cortex credit consumption across different warehouses.
*   **Detailed Analysis:**
    *   Explore usage by warehouse and model.
    *   Analyze model performance over time.
    *   View credit distribution by model.
*   **Cortex Services Analytics:** Dedicated sections for monitoring Cortex Analyst and Cortex Search usage and costs.
*   **Cost Optimization:** Receive actionable recommendations to improve cost efficiency.
*   **Interactive Filtering:** Dynamically filter the dashboard by date range and warehouses.

## 🛠️ Prerequisites

*   A Snowflake account with access to the `SNOWFLAKE.ACCOUNT_USAGE` schema. The role used to run the app will need permissions to view these tables.
*   Python 3.8+

## ⚙️ Installation

### Streamlit in Snowflake (SiS) - Recommended

This application is designed to run natively in Snowflake using Streamlit in Snowflake (SiS):

1.  **Create a new Streamlit app in Snowflake:**
    ```sql
    CREATE STREAMLIT llm_usage_analytics
    ROOT_LOCATION = '@<your_stage>/streamlit_app'
    MAIN_FILE = '/streamlit_app.py'
    QUERY_WAREHOUSE = '<your_warehouse>';
    ```

2.  **Upload the application file:**
    - Use the Snowflake web interface to upload `streamlit_app.py` to your designated stage
    - Or use SnowSQL to put the file:
    ```sql
    PUT file://streamlit_app.py @<your_stage>/streamlit_app/;
    ```

3.  **Grant necessary permissions:**
    Ensure your role has access to the `SNOWFLAKE.ACCOUNT_USAGE` schema:
    ```sql
    GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE TO ROLE <your_role>;
    ```

4.  **Launch the application:**
    Navigate to the Streamlit app in the Snowflake web interface or use:
    ```sql
    SHOW STREAMLITS;
    ```

### Local Development (Optional)

For local development and testing:

1.  **Install dependencies:**
    ```bash
    pip install streamlit pandas snowflake-snowpark-python plotly
    ```

2.  **Configure Snowflake connection:**
    Set up your connection parameters in a `connection.json` file or environment variables.

3.  **Run locally:**
    ```bash
    streamlit run streamlit_app.py
    ```

## 🚀 Usage

This application is designed to be run as a Snowflake Native App or in an environment with an active Snowflake session (e.g., Snowsight worksheet, Snowpark for Python). The `get_active_session()` function automatically retrieves the existing Snowflake session.

If you are running this locally and outside of a native Snowflake environment, you will need to configure your Snowflake connection credentials. Please refer to the [Snowflake documentation](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session) for setting up a local connection.

To run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application will open in your web browser.

## 📊 Data Source

This dashboard exclusively uses the `SNOWFLAKE.ACCOUNT_USAGE` schema, including the following views:
*   `METERING_HISTORY`
*   `WAREHOUSE_METERING_HISTORY`
*   `CORTEX_FUNCTIONS_USAGE_HISTORY`
*   `CORTEX_ANALYST_USAGE_HISTORY`
*   `CORTEX_SEARCH_SERVING_USAGE_HISTORY`

Data from these views can have a latency of up to a few hours.

## 🙌 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.
