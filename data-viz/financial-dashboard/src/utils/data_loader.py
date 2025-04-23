import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Union
import streamlit as st
import os
import re

def clean_json_content(content: str) -> str:
    """Clean JSON content to fix common formatting issues.
    
    Args:
        content (str): Raw JSON content
        
    Returns:
        str: Cleaned JSON content
    """
    # Remove any BOM characters
    content = content.strip('\ufeff')
    
    # Remove comments
    content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.S)
    
    # Fix trailing commas in arrays and objects
    content = re.sub(r',(\s*[\]}])', r'\1', content)
    
    # Replace NaN, Infinity, -Infinity with null
    # More comprehensive pattern to catch variations of NaN
    content = re.sub(r':\s*NaN\s*([,}])', r': null\1', content)
    content = re.sub(r':\s*-?Infinity\s*([,}])', r': null\1', content)
    
    # Fix unquoted property names
    content = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
    
    # Fix single quotes to double quotes
    content = re.sub(r"'([^']*)':", r'"\1":', content)  # Fix property names
    content = re.sub(r":\s*'([^']*)'", r':"\1"', content)  # Fix string values
    
    # Remove any invalid control characters
    content = ''.join(char for char in content if ord(char) >= 32 or char == '\n')
    
    # Additional cleanup for common issues
    # Handle empty values
    content = re.sub(r':\s*,', r': null,', content)
    content = re.sub(r':\s*}', r': null}', content)
    
    # Handle trailing decimal points
    content = re.sub(r'(\d+)\.\s*([,}])', r'\1.0\2', content)
    
    # Handle invalid URLs in keyStats
    content = re.sub(r'"keyStats":\s*"Error:.*?"', r'"keyStats": null', content)
    
    return content

def load_data(file_path):
    """Load fund data from JSON file with robust error handling.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary containing fund data or None if loading fails
    """
    try:
        # Try multiple possible paths
        possible_paths = [
            file_path,
            os.path.join(os.getcwd(), file_path),
            os.path.join(os.getcwd(), '..', file_path),
            os.path.abspath(file_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path)
        ]
        
        # Try each path until we find the file
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    try:
                        # First try direct JSON loading
                        funds_data = json.loads(content)
                    except json.JSONDecodeError:
                        # Clean the content and try again
                        cleaned_content = clean_json_content(content)
                        
                        try:
                            funds_data = json.loads(cleaned_content)
                        except json.JSONDecodeError as e:
                            # If still failing, try to fix the specific location of the error
                            line_no = int(str(e).split('line')[1].split()[0])
                            col_no = int(str(e).split('column')[1].split()[0])
                            
                            # Get the problematic line and surrounding context
                            lines = cleaned_content.split('\n')
                            problem_line = lines[line_no - 1] if line_no <= len(lines) else ""
                            
                            st.error(f"""JSON parsing error at line {line_no}, column {col_no}
                            Problem line: {problem_line}
                            Please check the JSON file for formatting issues around this location.""")
                            continue
                    
                    # Convert to dictionary with ISIN as key if needed
                    if isinstance(funds_data, list):
                        funds_dict = {}
                        for fund in funds_data:
                            if isinstance(fund, dict):
                                # Try different possible ISIN field names
                                isin = fund.get('ISIN') or fund.get('isin') or fund.get('Isin')
                                if isin:
                                    funds_dict[isin] = fund
                        return funds_dict
                    
                    return funds_data
        
        st.error(f"Data file not found in any of the expected locations. Tried: {possible_paths}")
        return None
            
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return None

def get_fund_data(funds_data, isin):
    """Get data for a specific fund by ISIN.
    
    Args:
        funds_data (dict): Dictionary containing all fund data
        isin (str): ISIN of the fund to retrieve
        
    Returns:
        dict: Fund data or None if not found
    """
    try:
        return funds_data.get(isin)
    except Exception as e:
        st.error(f"Error retrieving fund data: {str(e)}")
        return None

def get_all_funds_data(funds_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Get data for all funds.
    
    Args:
        funds_dict (Dict[str, Dict]): Dictionary of all funds
        
    Returns:
        Dict[str, Dict]: Copy of the funds dictionary
    """
    return funds_dict.copy()

def get_fund_metrics(fund_data: Dict) -> Dict[str, Union[float, str]]:
    """
    Extract key metrics from fund data.
    
    Args:
        fund_data (Dict): Fund data dictionary
        
    Returns:
        Dict[str, Union[float, str]]: Dictionary of key metrics
    """
    if not fund_data:
        return {}
    
    # Get ISIN and Product Name directly from the fund data
    metrics = {
        'ISIN': fund_data.get('ISIN') or fund_data.get('isin', 'N/A'),
        'Product Name': fund_data.get('name', 'N/A'),
        'Currency': fund_data.get('currency') or fund_data.get('Currency', 'N/A'),
        'Asset Class': fund_data.get('assetClass') or fund_data.get('AssetClass', 'N/A'),
        'Fund Size': fund_data.get('fundSize') or fund_data.get('FundSize', 'N/A'),
    }
    
    # Add performance metrics if available
    performance = fund_data.get('performanceMetrics', {})
    if performance:
        metrics.update({
            '1Y Return': performance.get('oneYearReturn', 'N/A'),
            '3Y Return': performance.get('threeYearReturn', 'N/A'),
            '5Y Return': performance.get('fiveYearReturn', 'N/A'),
            'YTD Return': performance.get('ytdReturn', 'N/A'),
        })
    
    return metrics