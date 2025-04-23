import pandas as pd

def calculate_net_cash_flows(data):
    cash_flows = []
    for entry in data:
        flows = entry.get('historicalData', {}).get('fundFlowData', [])
        total_flow = sum(flow['value'] for flow in flows if flow['value'] is not None)
        cash_flows.append({
            'ISIN': entry['ISIN'],
            'net_cash_flow': total_flow
        })
    return pd.DataFrame(cash_flows)

def extract_top_holdings(data, top_n=5):
    top_holdings = []
    for entry in data:
        holdings = entry.get('holdings', [])
        sorted_holdings = sorted(holdings, key=lambda x: x['weighting'], reverse=True)[:top_n]
        top_holdings.append({
            'ISIN': entry['ISIN'],
            'top_holdings': sorted_holdings
        })
    return top_holdings

def compute_financial_metrics(data):
    metrics = []
    for entry in data:
        financials = entry.get('financialMetrics', {}).get('fund', {})
        metrics.append({
            'ISIN': entry['ISIN'],
            'alpha': financials.get('alpha', None),
            'beta': financials.get('beta', None),
            'r_squared': financials.get('r_squared', None)
        })
    return pd.DataFrame(metrics)

def process_risk_metrics(risk_data):
    """
    Process risk metrics data from fundRiskVolatility, categoryRiskVolatility, or indexRiskVolatility.
    
    Args:
        risk_data (dict): Risk volatility data containing metrics for different time periods
        
    Returns:
        pd.DataFrame: Processed risk metrics in a DataFrame format
    """
    if not risk_data or not isinstance(risk_data, dict):
        return None
        
    periods = ['for1Year', 'for3Year', 'for5Year']
    metrics_list = ['alpha', 'beta', 'rSquared', 'standardDeviation', 'sharpeRatio']
    
    metrics = {}
    for period in periods:
        period_data = risk_data.get(period, {})
        if period_data and isinstance(period_data, dict):
            period_metrics = {}
            for metric in metrics_list:
                value = period_data.get(metric)
                # Handle None, NaN, and invalid values
                if value is not None and not pd.isna(value):
                    try:
                        value = float(value)
                        period_metrics[metric] = value
                    except (ValueError, TypeError):
                        period_metrics[metric] = None
                else:
                    period_metrics[metric] = None
            if any(v is not None for v in period_metrics.values()):
                metrics[period.replace('for', '')] = period_metrics
    
    if not metrics:
        return None
        
    df = pd.DataFrame(metrics).T
    df.columns = ['Alpha', 'Beta', 'R-Squared', 'Std Dev', 'Sharpe']
    df = df.round(3)
    
    return df