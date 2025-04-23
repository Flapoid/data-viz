import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import numpy as np
import logging
import tempfile
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from utils.data_processor import (
    calculate_net_cash_flows,
    extract_top_holdings,
    compute_financial_metrics,
    process_risk_metrics
)
from utils.data_loader import (
    load_data,
    get_fund_data,
    get_fund_metrics
)

def generate_pdf_report(funds_data, selected_isins, fund_options):
    """Generate a PDF report with all charts for selected funds."""
    # Create a temporary directory to store chart images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create PDF document
        pdf_path = os.path.join(temp_dir, "fund_report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Fund Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Process each selected fund
        for selected_isin in selected_isins:
            fund = funds_data.get(selected_isin)
            if not fund:
                continue
                
            fund_name = next((label.split(" - ")[1] for isin, label in fund_options if isin == selected_isin), selected_isin)
            
            # Add fund name as section header
            story.append(Paragraph(f"Fund: {fund_name}", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            # Performance Comparison Chart
            if 'historicalData' in fund and 'graphData' in fund['historicalData']:
                graph_data = fund['historicalData']['graphData']
                fund_df = pd.DataFrame(graph_data.get('fund', []))
                index_df = pd.DataFrame(graph_data.get('index', []))
                
                if not fund_df.empty or not index_df.empty:
                    fig = go.Figure()
                    
                    # Add fund data
                    if not fund_df.empty:
                        fund_df['date'] = pd.to_datetime(fund_df['date'])
                        today = pd.Timestamp.now().normalize()
                        fund_df = fund_df[fund_df['date'] <= today]
                        fig.add_trace(go.Scatter(
                            x=fund_df['date'],
                            y=fund_df['value'],
                            name='Fund',
                            mode='lines'
                        ))
                    
                    # Add index data
                    if not index_df.empty:
                        index_df['date'] = pd.to_datetime(index_df['date'])
                        index_df = index_df[index_df['date'] <= today]
                        fig.add_trace(go.Scatter(
                            x=index_df['date'],
                            y=index_df['value'],
                            name='Index',
                            mode='lines'
                        ))
                    
                    fig.update_layout(
                        title=f'{fund_name} Performance',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        hovermode='x unified'
                    )
                    
                    # Save chart as image
                    chart_path = os.path.join(temp_dir, f"performance_{selected_isin}.png")
                    fig.write_image(chart_path)
                    
                    # Add chart to PDF
                    story.append(Paragraph("Performance Chart", styles['Heading3']))
                    story.append(Image(chart_path, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 20))
            
            # Add holdings chart if available
            if 'holdings' in fund:
                holdings_df = pd.DataFrame(fund['holdings'])
                if not holdings_df.empty:
                    fig = px.pie(holdings_df, values='weighting', names='securityName', title='Top Holdings')
                    chart_path = os.path.join(temp_dir, f"holdings_{selected_isin}.png")
                    fig.write_image(chart_path)
                    
                    story.append(Paragraph("Holdings Distribution", styles['Heading3']))
                    story.append(Image(chart_path, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 20))
            
            # Add page break between funds
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        
        # Return the PDF file path
        return pdf_path

# Suppress Streamlit warnings about ScriptRunContext
streamlit_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('streamlit')]
for logger in streamlit_loggers:
    logger.setLevel(logging.ERROR)

def format_metric(value):
    """Format a metric value for display.
    
    Args:
        value: The value to format (can be None, NaN, or a number)
        
    Returns:
        str: Formatted value or 'N/A' if value is None or NaN
    """
    if value is None or pd.isna(value):
        return 'N/A'
    try:
        # Format as percentage if the value is between -1 and 1
        if -1 <= value <= 1:
            return f"{value:.2%}"
        # Format as decimal with 2 decimal places otherwise
        return f"{value:.2f}"
    except (TypeError, ValueError):
        return 'N/A'

# Set page config
st.set_page_config(
    page_title="Fund Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

def create_price_chart(historical_data):
    """Create a line chart showing price performance over time for fund, index and category."""
    try:
        if not historical_data or 'graphData' not in historical_data:
            return None
            
        graph_data = historical_data['graphData']
        
        # Create DataFrames for each series
        fund_df = pd.DataFrame(graph_data.get('fund', []))
        index_df = pd.DataFrame(graph_data.get('index', []))
        category_df = pd.DataFrame(graph_data.get('category', []))
        
        # Convert dates to datetime
        fund_df['date'] = pd.to_datetime(fund_df['date'])
        index_df['date'] = pd.to_datetime(index_df['date'])
        category_df['date'] = pd.to_datetime(category_df['date'])
        
        # Filter data up to today's date
        today = pd.Timestamp.now().normalize()
        fund_df = fund_df[fund_df['date'] <= today]
        index_df = index_df[index_df['date'] <= today]
        category_df = category_df[category_df['date'] <= today]
        
        # Create figure with three traces
        fig = go.Figure()
        
        # Add fund trace
        fig.add_trace(go.Scatter(
            x=fund_df['date'],
            y=fund_df['value'],
            name='Fund',
            line=dict(color='blue', width=2)
        ))
        
        # Add index trace
        fig.add_trace(go.Scatter(
            x=index_df['date'],
            y=index_df['value'],
            name='Index',
            line=dict(color='red', width=2)
        ))
        
        # Add category trace
        fig.add_trace(go.Scatter(
            x=category_df['date'],
            y=category_df['value'],
            name='Category',
            line=dict(color='green', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Price Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            xaxis=dict(
                range=[min(fund_df['date'].min(), index_df['date'].min(), category_df['date'].min()), today]
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        return None

def create_flows_chart(flows_data):
    try:
        if not flows_data or len(flows_data) == 0:
            return None
        
        df = pd.DataFrame(flows_data)
        df['date'] = pd.to_datetime(df['date'])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['amount'],
            marker_color=np.where(df['amount'] >= 0, 'green', 'red')
        ))
        fig.update_layout(title='Net Flows')
        return fig
    except Exception as e:
        st.error(f"Error creating flows chart: {str(e)}")
        return None

def create_holdings_chart(holdings_data):
    try:
        if not holdings_data or len(holdings_data) == 0:
            return None
        
        df = pd.DataFrame(holdings_data)
        fig = px.pie(df, values='weighting', names='securityName', title='Top Holdings')
        return fig
    except Exception as e:
        st.error(f"Error creating holdings chart: {str(e)}")
        return None

def create_allocation_chart(allocation_data):
    try:
        if not allocation_data:
            return None
        
        # Extract style box data
        style_data = {
            'Large Value': float(allocation_data.get('largeValue', 0)),
            'Large Blend': float(allocation_data.get('largeBlend', 0)),
            'Large Growth': float(allocation_data.get('largeGrowth', 0)),
            'Mid Value': float(allocation_data.get('middleValue', 0)),
            'Mid Blend': float(allocation_data.get('middleBlend', 0)),
            'Mid Growth': float(allocation_data.get('middleGrowth', 0)),
            'Small Value': float(allocation_data.get('smallValue', 0)),
            'Small Blend': float(allocation_data.get('smallBlend', 0)),
            'Small Growth': float(allocation_data.get('smallGrowth', 0))
        }
        
        df = pd.DataFrame(list(style_data.items()), columns=['Style', 'Weight'])
        fig = px.bar(df, x='Style', y='Weight', title='Style Box Allocation')
        return fig
    except Exception as e:
        st.error(f"Error creating allocation chart: {str(e)}")
        return None

def create_risk_metrics_chart(fund_data):
    """Create a chart for risk metrics."""
    try:
        if not fund_data or 'fundRiskVolatility' not in fund_data:
            st.warning("Risk metrics data not available for this fund.")
            return

        risk_data = fund_data['fundRiskVolatility']
        
        # Create columns for different time periods
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1 Year")
            if 'for1Year' in risk_data:
                year1 = risk_data['for1Year']
                st.metric("Alpha", format_metric(year1.get('alpha')))
                st.metric("Beta", format_metric(year1.get('beta')))
                st.metric("R-Squared", format_metric(year1.get('rSquared')))
            else:
                st.warning("1 Year data not available")
        
        with col2:
            st.subheader("3 Years")
            if 'for3Year' in risk_data:
                year3 = risk_data['for3Year']
                st.metric("Alpha", format_metric(year3.get('alpha')))
                st.metric("Beta", format_metric(year3.get('beta')))
                st.metric("R-Squared", format_metric(year3.get('rSquared')))
            else:
                st.warning("3 Year data not available")
        
        with col3:
            st.subheader("5 Years")
            if 'for5Year' in risk_data:
                year5 = risk_data['for5Year']
                st.metric("Alpha", format_metric(year5.get('alpha')))
                st.metric("Beta", format_metric(year5.get('beta')))
                st.metric("R-Squared", format_metric(year5.get('rSquared')))
            else:
                st.warning("5 Year data not available")
                
    except Exception as e:
        st.error(f"Error creating risk metrics chart: {str(e)}")

def process_risk_metrics(risk_data):
    """Process risk metrics data and return a formatted DataFrame.
    
    Args:
        risk_data (dict): Dictionary containing risk metrics data from riskVolatility
        
    Returns:
        pd.DataFrame: Formatted DataFrame with risk metrics or None if data is invalid
    """
    if not risk_data:
        return None
        
    try:
        metrics = {}
        
        # Process each time period
        for period in ['for1Year', 'for3Year', 'for5Year']:
            if period in risk_data:
                period_data = risk_data[period]
                if period_data and isinstance(period_data, dict):  # Check if period_data is a non-empty dict
                    metrics[period] = {
                        'Alpha': format_metric(period_data.get('alpha')),
                        'Beta': format_metric(period_data.get('beta')),
                        'R-Squared': format_metric(period_data.get('rSquared')),
                        'Standard Deviation': format_metric(period_data.get('standardDeviation')),
                        'Sharpe Ratio': format_metric(period_data.get('sharpeRatio'))
                    }
        
        if not metrics:
            return None
            
        # Create DataFrame
        df = pd.DataFrame(metrics).T
        df.index = ['1 Year', '3 Years', '5 Years']
        return df
        
    except Exception as e:
        st.error(f"Error processing risk metrics: {str(e)}")
        return None

def create_net_assets_chart(graph_data):
    """Create a line chart showing net assets over time."""
    try:
        if not graph_data or 'data' not in graph_data:
            return None
        
        # Create DataFrame with quarterly and yearly data
        data = []
        for year_data in graph_data['data']:
            year = year_data['yr']
            # Add quarterly data
            for q in range(1, 5):
                quarter = f"Q{q}"
                na_value = year_data.get(f'na{quarter}')
                if na_value is not None:
                    data.append({
                        'Date': f"{year}-{q*3:02d}-01",
                        'Value': na_value,
                        'Type': 'Quarterly'
                    })
            # Add yearly data
            if year_data.get('naYr') is not None:
                data.append({
                    'Date': f"{year}-12-31",
                    'Value': year_data['naYr'],
                    'Type': 'Yearly'
                })
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter data up to today's date
        today = pd.Timestamp.now().normalize()
        df = df[df['Date'] <= today]
        
        # Sort by date to ensure chronological order
        df = df.sort_values('Date')
        
        fig = go.Figure()
        
        # Add quarterly line
        quarterly_data = df[df['Type'] == 'Quarterly']
        fig.add_trace(go.Scatter(
            x=quarterly_data['Date'],
            y=quarterly_data['Value'],
            name='Quarterly Net Assets',
            line=dict(color='blue', width=2)
        ))
        
        # Add yearly markers
        yearly_data = df[df['Type'] == 'Yearly']
        fig.add_trace(go.Scatter(
            x=yearly_data['Date'],
            y=yearly_data['Value'],
            name='Yearly Net Assets',
            mode='markers',
            marker=dict(size=8, color='red')
        ))
        
        fig.update_layout(
            title='Net Assets Over Time',
            xaxis_title='Date',
            yaxis_title='Net Assets (Bil)',
            hovermode='x unified',
            xaxis=dict(
                range=[df['Date'].min(), today]  # Set x-axis range to include today
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating net assets chart: {str(e)}")
        return None

def create_net_flows_chart(graph_data):
    """Create a bar chart showing net flows over time."""
    try:
        if not graph_data or 'data' not in graph_data:
            return None
        
        # Create DataFrame with quarterly and yearly data
        data = []
        for year_data in graph_data['data']:
            year = year_data['yr']
            # Add quarterly data
            for q in range(1, 5):
                quarter = f"Q{q}"
                nf_value = year_data.get(f'nf{quarter}')
                if nf_value is not None:
                    data.append({
                        'Date': f"{year}-{q*3:02d}-01",
                        'Value': nf_value,
                        'Type': 'Quarterly'
                    })
            # Add yearly data
            if year_data.get('nfYr') is not None:
                data.append({
                    'Date': f"{year}-12-31",
                    'Value': year_data['nfYr'],
                    'Type': 'Yearly'
                })
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter data up to today's date
        today = pd.Timestamp.now().normalize()
        df = df[df['Date'] <= today]
        
        # Sort by date to ensure chronological order
        df = df.sort_values('Date')
        
        fig = go.Figure()
        
        # Add quarterly bars
        quarterly_data = df[df['Type'] == 'Quarterly']
        fig.add_trace(go.Bar(
            x=quarterly_data['Date'],
            y=quarterly_data['Value'],
            name='Quarterly Net Flows',
            marker_color=np.where(quarterly_data['Value'] >= 0, 'green', 'red')
        ))
        
        # Add yearly markers
        yearly_data = df[df['Type'] == 'Yearly']
        fig.add_trace(go.Scatter(
            x=yearly_data['Date'],
            y=yearly_data['Value'],
            name='Yearly Net Flows',
            mode='markers',
            marker=dict(size=8, color='black')
        ))
        
        fig.update_layout(
            title='Net Flows Over Time',
            xaxis_title='Date',
            yaxis_title='Net Flows (Bil)',
            hovermode='x unified',
            barmode='group',
            xaxis=dict(
                range=[df['Date'].min(), today]  # Set x-axis range to include today
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating net flows chart: {str(e)}")
        return None

def get_fund_metrics(fund):
    """Get fund metrics including performance data from historical data."""
    metrics = {}
    
    # Basic fund info
    metrics['ISIN'] = fund.get('isin', 'N/A')
    metrics['Product Name'] = fund.get('productName', 'N/A')
    
    # Get currency from historical data
    historical_data = fund.get('historicalData', {})
    metrics['Currency'] = historical_data.get('baseCurrency', 'N/A')
    
    # Get performance metrics from trailingReturn data
    trailing_return = fund.get('trailingReturn', {})
    if trailing_return and 'totalReturnNAV' in trailing_return:
        column_defs = trailing_return.get('columnDefs', [])
        return_values = trailing_return.get('totalReturnNAV', [])
        
        # Create a mapping of period to return value
        returns_map = dict(zip(column_defs, return_values))
        
        # Map the returns to our metrics
        metrics['1Y Return'] = returns_map.get('1Year')
        metrics['3Y Return'] = returns_map.get('3Year')
        metrics['5Y Return'] = returns_map.get('5Year')
        metrics['YTD Return'] = returns_map.get('YearToDate')
    else:
        # If trailingReturn data is not available, set all returns to None
        metrics.update({
            '1Y Return': None,
            '3Y Return': None,
            '5Y Return': None,
            'YTD Return': None
        })
    
    return metrics

def main():
    st.title("Fund Analysis Dashboard")
    
    # Load data with correct path
    funds_data = load_data('data-viz/financial-dashboard/data/funds_results.json')
    
    if funds_data is None:
        st.error("Failed to load fund data. Please check the data file.")
        return
    
    if not funds_data:
        st.error("No funds data available. Please check your data file.")
        return
    
    # Create fund options list with ISIN and Product Name
    fund_options = [(isin, f"{isin} - {fund.get('Product Name', 'Unknown Fund')}")
                   for isin, fund in funds_data.items()]
    
    if not fund_options:
        st.error("No funds available for selection.")
        return
        
    selected_isins = st.multiselect(
        "Select Funds",
        options=[isin for isin, _ in fund_options],
        format_func=lambda x: next((label for isin, label in fund_options if isin == x), x)
    )
    
    if selected_isins:
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Performance", "Holdings", "Risk Metrics", "Style Box"])
        
        with tab1:
            st.header("Fund Overview")
            
            # Create a metric container for each selected fund
            for selected_isin in selected_isins:
                fund = funds_data.get(selected_isin)
                if not fund:
                    st.error(f"No data found for ISIN: {selected_isin}")
                    continue
                    
                # Get fund metrics
                metrics = get_fund_metrics(fund)
                
                # Create a subheader for each fund
                st.subheader(next((label.split(" - ")[1] for isin, label in fund_options if isin == selected_isin), "N/A"))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ISIN", selected_isin)
                with col2:
                    st.metric("Product Name", next((label.split(" - ")[1] for isin, label in fund_options if isin == selected_isin), "N/A"))
                with col3:
                    st.metric("Currency", metrics.get('Currency', 'N/A'))
                
                # Display performance metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    value = metrics.get('1Y Return')
                    st.metric("1 Year Return", f"{value:.2f}%" if value is not None else 'N/A')
                with col2:
                    value = metrics.get('3Y Return')
                    st.metric("3 Year Return", f"{value:.2f}%" if value is not None else 'N/A')
                with col3:
                    value = metrics.get('5Y Return')
                    st.metric("5 Year Return", f"{value:.2f}%" if value is not None else 'N/A')
                with col4:
                    value = metrics.get('YTD Return')
                    st.metric("YTD Return", f"{value:.2f}%" if value is not None else 'N/A')
                
                st.divider()  # Add a visual separator between funds
        
        with tab2:
            st.header("Performance Analysis")
            
            # Create a combined performance chart for all selected funds
            st.subheader("Fund-to-Fund Comparison")
            fig = go.Figure()
            
            for selected_isin in selected_isins:
                fund = funds_data.get(selected_isin)
                if not fund or 'historicalData' not in fund or 'graphData' not in fund['historicalData']:
                    continue
                
                graph_data = fund['historicalData']['graphData']
                fund_df = pd.DataFrame(graph_data.get('fund', []))
                
                if not fund_df.empty:
                    fund_df['date'] = pd.to_datetime(fund_df['date'])
                    today = pd.Timestamp.now().normalize()
                    fund_df = fund_df[fund_df['date'] <= today]
                    
                    fund_name = next((label.split(" - ")[1] for isin, label in fund_options if isin == selected_isin), selected_isin)
                    fig.add_trace(go.Scatter(
                        x=fund_df['date'],
                        y=fund_df['value'],
                        name=fund_name,
                        mode='lines'
                    ))
            
            if len(fig.data) > 0:
                fig.update_layout(
                    title='Fund Performance Comparison',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True, key='fund_performance_comparison')
            else:
                st.warning("No price performance data available for selected funds")
            
            # Display individual fund vs index charts
            st.subheader("Individual Fund vs Index Comparisons")
            
            for selected_isin in selected_isins:
                fund = funds_data.get(selected_isin)
                if not fund:
                    continue
                
                fund_name = next((label.split(" - ")[1] for isin, label in fund_options if isin == selected_isin), selected_isin)
                st.write(f"### {fund_name}")
                
                # Create fund vs index comparison chart
                if 'historicalData' in fund and 'graphData' in fund['historicalData']:
                    graph_data = fund['historicalData']['graphData']
                    fund_df = pd.DataFrame(graph_data.get('fund', []))
                    index_df = pd.DataFrame(graph_data.get('index', []))
                    
                    if not fund_df.empty or not index_df.empty:
                        fig = go.Figure()
                        
                        # Add fund data
                        if not fund_df.empty:
                            fund_df['date'] = pd.to_datetime(fund_df['date'])
                            today = pd.Timestamp.now().normalize()
                            fund_df = fund_df[fund_df['date'] <= today]
                            fig.add_trace(go.Scatter(
                                x=fund_df['date'],
                                y=fund_df['value'],
                                name='Fund',
                                mode='lines'
                            ))
                        
                        # Add index data
                        if not index_df.empty:
                            index_df['date'] = pd.to_datetime(index_df['date'])
                            index_df = index_df[index_df['date'] <= today]
                            fig.add_trace(go.Scatter(
                                x=index_df['date'],
                                y=index_df['value'],
                                name='Index',
                                mode='lines'
                            ))
                        
                        fig.update_layout(
                            title=f'{fund_name} vs Index Performance',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f'fund_vs_index_{selected_isin}')
                    else:
                        st.info("No performance comparison data available")
                else:
                    st.info("No historical data available for comparison")
                
                # Get graph data for net assets and flows
                graph_data = fund.get('graphData', {})
                
                # Create and display net assets chart
                net_assets_fig = create_net_assets_chart(graph_data)
                if net_assets_fig:
                    st.plotly_chart(net_assets_fig, use_container_width=True, key=f'net_assets_{selected_isin}')
                
                # Create and display net flows chart
                net_flows_fig = create_net_flows_chart(graph_data)
                if net_flows_fig:
                    st.plotly_chart(net_flows_fig, use_container_width=True, key=f'net_flows_{selected_isin}')
                
                st.divider()
        
        with tab3:
            st.header("Holdings Analysis")
            
            for selected_isin in selected_isins:
                fund = funds_data.get(selected_isin)
                if not fund:
                    continue
                    
                fund_name = next((label.split(" - ")[1] for isin, label in fund_options if isin == selected_isin), selected_isin)
                st.subheader(f"{fund_name} - Holdings")
                
                # Top holdings pie chart
                if 'holdings' in fund:
                    fig = create_holdings_chart(fund['holdings'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display holdings table
                    holdings_df = pd.DataFrame(fund['holdings'])
                    if not holdings_df.empty:
                        st.dataframe(holdings_df)
                else:
                    st.info("No holdings data available")
                
                st.divider()
        
        with tab4:
            st.header("Risk Metrics")
            
            for selected_isin in selected_isins:
                fund = funds_data.get(selected_isin)
                if not fund:
                    continue
                    
                fund_name = next((label.split(" - ")[1] for isin, label in fund_options if isin == selected_isin), selected_isin)
                st.subheader(f"{fund_name} - Risk Metrics")
                
                # Process risk metrics
                fund_risk = process_risk_metrics(fund.get('riskVolatility', {}).get('fundRiskVolatility', {}))
                category_risk = process_risk_metrics(fund.get('riskVolatility', {}).get('categoryRiskVolatility', {}))
                index_risk = process_risk_metrics(fund.get('riskVolatility', {}).get('indexRiskVolatility', {}))
                
                if fund_risk is not None or category_risk is not None or index_risk is not None:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("Fund Risk Metrics")
                        if fund_risk is not None:
                            st.dataframe(fund_risk, use_container_width=True)
                        else:
                            st.info("No fund risk metrics available")
                    
                    with col2:
                        st.write("Category Risk Metrics")
                        if category_risk is not None:
                            st.dataframe(category_risk, use_container_width=True)
                        else:
                            st.info("No category risk metrics available")
                    
                    with col3:
                        st.write("Index Risk Metrics")
                        if index_risk is not None:
                            st.dataframe(index_risk, use_container_width=True)
                        else:
                            st.info("No index risk metrics available")
                else:
                    st.warning("No risk metrics data available for this fund")
                
                st.divider()
        
        with tab5:
            st.header("Style Box Analysis")
            
            for selected_isin in selected_isins:
                fund = funds_data.get(selected_isin)
                if not fund:
                    continue
                    
                fund_name = next((label.split(" - ")[1] for isin, label in fund_options if isin == selected_isin), selected_isin)
                st.subheader(f"{fund_name} - Style Box")
                
                if 'allocationWeighting' in fund:
                    fig = create_allocation_chart(fund['allocationWeighting'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No style box data available")
                
                st.divider()
    else:
        st.info("Please select one or more funds to analyze")

if __name__ == "__main__":
    main()
