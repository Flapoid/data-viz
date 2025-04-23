from typing import List
import pandas as pd
import streamlit as st

def display_top_holdings(holdings: List[dict]):
    if not holdings:
        st.write("No holdings data available.")
        return

    holdings_df = pd.DataFrame(holdings)
    top_holdings = holdings_df[['securityName', 'weighting', 'marketValue']].sort_values(by='weighting', ascending=False)

    st.subheader("Top Holdings by Allocation")
    st.table(top_holdings)

def display_cash_flows(cash_flows: List[dict]):
    if not cash_flows:
        st.write("No cash flow data available.")
        return

    cash_flows_df = pd.DataFrame(cash_flows)
    st.subheader("Net Cash Flows")
    st.bar_chart(cash_flows_df['value'])

def display_historical_data(historical_data: List[dict]):
    if not historical_data:
        st.write("No historical data available.")
        return

    historical_df = pd.DataFrame(historical_data)
    st.subheader("Historical Data")
    st.line_chart(historical_df['value'])  # Assuming 'value' is a column in historical data

def display_financial_metrics(metrics: dict):
    st.subheader("Financial Metrics")
    for key, value in metrics.items():
        st.write(f"{key}: {value}")