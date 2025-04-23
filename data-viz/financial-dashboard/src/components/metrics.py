from scipy import stats
import pandas as pd

def calculate_alpha_beta_r_squared(returns, benchmark_returns):
    # Calculate beta
    beta, alpha = stats.linregress(benchmark_returns, returns)[:2]
    
    # Calculate R-squared
    r_squared = stats.linregress(benchmark_returns, returns)[2] ** 2
    
    return alpha, beta, r_squared

def get_financial_metrics(data):
    metrics = {}
    for fund in data:
        fund_returns = fund['historicalData']['fundFlowData']  # Placeholder for actual returns data
        benchmark_returns = fund['historicalData']['fundFlowData']  # Placeholder for benchmark returns
        
        alpha, beta, r_squared = calculate_alpha_beta_r_squared(fund_returns, benchmark_returns)
        
        metrics[fund['ISIN']] = {
            'Alpha': alpha,
            'Beta': beta,
            'R-squared': r_squared
        }
    
    return metrics

def display_metrics(metrics):
    metrics_df = pd.DataFrame(metrics).T
    return metrics_df