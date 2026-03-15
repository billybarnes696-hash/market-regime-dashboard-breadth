# app.py - Streamlit Breadth Confluence Dashboard (Improved Extraction)
import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
import time
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Breadth Confluence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ========================
# IMPROVED DATA FETCHING FUNCTIONS
# ========================

def fetch_stockcharts_page(url, page_num, retries=3):
    """
    Enhanced fetch with retries, better headers, and response validation.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }
    
    for attempt in range(retries):
        try:
            time.sleep(2)  # Rate limiting
            response = requests.get(url, headers=headers, timeout=45)
            response.raise_for_status()
            
            # Check if response contains expected content
            if len(response.text) < 10000:  # StockCharts pages are typically large
                logger.warning(f"Page {page_num} response unusually small ({len(response.text)} chars)")
                continue
                
            return response.text
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on page {page_num}, attempt {attempt+1}/{retries}")
            time.sleep(3)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on page {page_num}: {e}")
            time.sleep(3)
    
    return None


def extract_indicator_value(html_content, indicator_name, timeframe='Weekly'):
    """
    Enhanced extraction using patterns observed in actual StockCharts OCR output.
    
    Key patterns from PDF analysis:
    - $INDICATOR(Weekly] VALUE or $INDICATOR(Daily] VALUE
    - Values can be: positive/negative, with/without commas, varying decimals
    - Some indicators use % suffix, others don't
    """
    if not html_content:
        return None
    
    # Normalize whitespace and common OCR artifacts
    cleaned = re.sub(r'\s+', ' ', html_content)
    cleaned = re.sub(r'[^\x00-\x7F]+', '', cleaned)  # Remove non-ASCII
    
    # Indicator-specific patterns based on actual StockCharts output format
    patterns = {
        # Bullish Percent Indices (0-100 scale, often with % implied)
        'BPSPX': [
            r'\$BPSPX\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([-]?\d+\.?\d*)',
            r'\$BPSPX[^\d\-]*([-]?\d+\.?\d*)\s*(?:%|Chg|$)',
        ],
        'BPNYA': [
            r'\$BPNYA\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([-]?\d+\.?\d*)',
            r'\$BPNYA[^\d\-]*([-]?\d+\.?\d*)\s*(?:%|Chg|$)',
        ],
        
        # Momentum Oscillators (can be negative, wide ranges)
        'NYMO': [
            r'\$NYMO\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([-]?\d+\.?\d*)',
            r'\$NYMO[^\d\-]*([-]?\d+\.?\d*)',
        ],
        'NYSI': [
            r'\$NYSI\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([-]?\d+\.?\d*)',
            r'\$NYSI[^\d\-]*([-]?\d+\.?\d*)',
        ],
        
        # Percent Above Moving Average (0-100 scale)
        'SPXA50R': [
            r'\$SPXA50R\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'\$SPXA50R[^\d]*([\d.]+)\s*(?:%|$)',
        ],
        'OEXA50R': [
            r'\$OEXA50R\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'\$OEXA50R[^\d]*([\d.]+)\s*(?:%|$)',
        ],
        'OEXA150R': [
            r'\$OEXA150R\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'\$OEXA150R[^\d]*([\d.]+)\s*(?:%|$)',
        ],
        'OEXA200R': [
            r'\$OEXA200R\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'\$OEXA200R[^\d]*([\d.]+)\s*(?:%|$)',
        ],
        
        # Put/Call Ratio (0-2 range typically)
        'CPCE': [
            r'\$CPCE\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'\$CPCE[^\d]*([\d.]+)',
        ],
        
        # Volatility/VIX products
        'VXX': [
            r'VXX\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'VXX[^\d]*([\d.]+)',
        ],
        
        # Ratio indicators (typically 0.20-0.70 range)
        'RSP:SPY': [
            r'RSP:SPY\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'RSP:SPY[^\d]*([\d.]+)',
        ],
        'IWM:SPY': [
            r'IWM:SPY\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'IWM:SPY[^\d]*([\d.]+)',
        ],
        'SMH:SPY': [
            r'SMH:SPY\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'SMH:SPY[^\d]*([\d.]+)',
        ],
        'XLF:SPY': [
            r'XLF:SPY\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'XLF:SPY[^\d]*([\d.]+)',
        ],
        
        # Credit spreads (HYG ratios)
        'HYG:IEF': [
            r'HYG:IEF\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'HYG:IEF[^\d]*([\d.]+)',
        ],
        'HYG:TLT': [
            r'HYG:TLT\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'HYG:TLT[^\d]*([\d.]+)',
        ],
        
        # Leveraged/inverse products
        'SPXS:SVOL': [
            r'SPXS:SVOL\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'SPXS:SVOL[^\d]*([\d.]+)',
        ],
        'URSP': [
            r'URSP\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d.]+)',
            r'URSP[^\d]*([\d.]+)',
        ],
        
        # Price data (can have commas for large numbers)
        'RSP': [
            r'\bRSP\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d,]+\.?\d*)',
            r'\bRSP[^\d]*([\d,]+\.?\d*)\s*(?:Chg|Volume|$)',
        ],
        'SPX': [
            r'\$SPX\s*[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([\d,]+\.?\d*)',
            r'\$SPX[^\d]*([\d,]+\.?\d*)\s*(?:Chg|Volume|$)',
        ],
        
        # Cumulative indicators (can be very large, positive or negative)
        'NYAD': [
            r'\$NYAD\s*(?:Cumulative\s*)?[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([-]?\d+(?:,\d+)*(?:\.\d+)?)',
            r'\$NYAD[^\d\-]*([-]?\d+(?:,\d+)*(?:\.\d+)?)',
        ],
        'NYHL': [
            r'\$NYHL\s*(?:Cumulative\s*)?[\(\[]?(?:Weekly|Daily)[\]\)]?\s*[:\s]*([-]?\d+(?:,\d+)*(?:\.\d+)?)',
            r'\$NYHL[^\d\-]*([-]?\d+(?:,\d+)*(?:\.\d+)?)',
        ],
    }
    
    if indicator_name not in patterns:
        return None
    
    # Try each pattern for this indicator
    for pattern in patterns[indicator_name]:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            try:
                raw_value = match.group(1).replace(',', '')
                value = float(raw_value)
                
                # Basic sanity checks based on expected ranges
                if not validate_indicator_value(indicator_name, value):
                    logger.debug(f"Value {value} for {indicator_name} failed validation")
                    continue
                    
                return value
            except (ValueError, IndexError) as e:
                logger.debug(f"Parse error for {indicator_name}: {e}")
                continue
    
    return None


def validate_indicator_value(indicator_name, value):
    """
    Basic sanity checks to filter out obviously incorrect extractions.
    Adjust ranges based on historical data for each indicator.
    """
    validation_rules = {
        'BPSPX': (0, 100),
        'BPNYA': (0, 100),
        'NYMO': (-200, 200),
        'NYSI': (-2000, 2000),
        'SPXA50R': (0, 100),
        'OEXA50R': (0, 100),
        'OEXA150R': (0, 100),
        'OEXA200R': (0, 100),
        'CPCE': (0.1, 3.0),
        'VXX': (5, 200),
        'RSP:SPY': (0.15, 0.50),
        'HYG:IEF': (0.70, 0.95),
        'HYG:TLT': (0.80, 1.10),
        'IWM:SPY': (0.25, 0.55),
        'SMH:SPY': (0.30, 0.80),
        'XLF:SPY': (0.05, 0.12),
        'SPXS:SVOL': (0.5, 20),
        'URSP': (10, 100),
        'RSP': (50, 300),
        'SPX': (2000, 10000),
        'NYAD': (-100000, 300000),
        'NYHL': (-5000, 5000),
    }
    
    if indicator_name in validation_rules:
        min_val, max_val = validation_rules[indicator_name]
        return min_val <= value <= max_val
    return True  # No validation rule = accept value


def extract_all_indicators(html_content):
    """
    Extract all breadth indicators from combined HTML content.
    Returns dict with indicator names as keys and float values (or None).
    """
    indicators = [
        'BPSPX', 'BPNYA', 'NYMO', 'NYSI', 'SPXA50R', 'OEXA50R', 
        'OEXA150R', 'OEXA200R', 'CPCE', 'VXX', 'RSP:SPY', 
        'HYG:IEF', 'HYG:TLT', 'IWM:SPY', 'SMH:SPY', 'XLF:SPY',
        'SPXS:SVOL', 'URSP', 'RSP', 'SPX', 'NYAD', 'NYHL'
    ]
    
    results = {}
    for ind in indicators:
        results[ind] = extract_indicator_value(html_content, ind)
        if results[ind] is not None:
            logger.info(f"✓ Extracted {ind}: {results[ind]}")
        else:
            logger.debug(f"✗ Could not extract {ind}")
    
    return results


# [Rest of your existing functions remain the same: calculate_breadth_score, 
# get_signal_emoji, main(), etc.]
# Just replace the old extract_indicator_value and fetch functions above.
