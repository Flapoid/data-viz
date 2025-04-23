from typing import List, Dict
import pandas as pd
import plotly.express as px

def plot_price_performance(data: List[Dict], fund_name: str) -> None:
    df = pd.DataFrame(data)
    df['portfolioDate'] = pd.to_datetime(df['portfolioDate'])
    fund_data = df[df['equityStyle']['fund']['name'] == fund_name]

    fig = px.line(fund_data, x='portfolioDate', y='graphData.data', title=f'Price Performance of {fund_name}')
    fig.update_layout(xaxis_title='Date', yaxis_title='Price')
    fig.show()

def plot_net_cash_flows(data: List[Dict], fund_name: str) -> None:
    df = pd.DataFrame(data)
    fund_data = df[df['equityStyle']['fund']['name'] == fund_name]

    cash_flows = fund_data['historicalData']['fundFlowData']
    cash_flows_df = pd.DataFrame(cash_flows)

    fig = px.bar(cash_flows_df, x=cash_flows_df.index, y='value', title=f'Net Cash Flows for {fund_name}')
    fig.update_layout(xaxis_title='Time Period', yaxis_title='Net Cash Flow')
    fig.show()

def plot_top_holdings(data: List[Dict], fund_name: str) -> None:
    df = pd.DataFrame(data)
    fund_data = df[df['equityStyle']['fund']['name'] == fund_name]

    holdings = fund_data['holdings']
    holdings_df = pd.DataFrame(holdings)

    fig = px.pie(holdings_df, names='securityName', values='weighting', title=f'Top Holdings for {fund_name}')
    fig.show()