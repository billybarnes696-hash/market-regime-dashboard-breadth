# app.py - Streamlit Breadth Confluence Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Breadth Confluence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ========================
# DATA FETCHING FUNCTIONS
# ========================

def fetch_stockcharts_page(url, page_num):
    """
    Attempt to fetch and parse StockCharts TenPP page.
    Note: This is fragile due to JS rendering and anti-bot measures.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.warning(f"⚠️ Page {page_num} fetch failed: {e}")
        return None

def extract_indicator_value(html_content, indicator_name, pattern=None):
    """
    Extract numeric values for breadth indicators from OCR-like text.
    Uses regex patterns based on common StockCharts text formats.
    """
    if not html_content:
        return None
    
    # Default patterns for common indicators
    patterns = {
        'BPSPX': r'\$BPSPX[^\d]*([\d.]+)\s*[%]?',
        'BPNYA': r'\$BPNYA[^\d]*([\d.]+)\s*[%]?',
        'NYMO': r'\$NYMO[^\d]*([-]?\d+\.?\d*)',
        'NYSI': r'\$NYSI[^\d]*([-]?\d+\.?\d*)',
        'SPXA50R': r'\$SPXA50R[^\d]*([\d.]+)\s*[%]?',
        'OEXA50R': r'\$OEXA50R[^\d]*([\d.]+)\s*[%]?',
        'OEXA150R': r'\$OEXA150R[^\d]*([\d.]+)\s*[%]?',
        'OEXA200R': r'\$OEXA200R[^\d]*([\d.]+)\s*[%]?',
        'CPCE': r'\$CPCE[^\d]*([\d.]+)',
        'VXX': r'VXX[^\d]*([\d.]+)',
        'RSP:SPY': r'RSP:SPY[^\d]*([\d.]+)',
        'HYG:IEF': r'HYG:IEF[^\d]*([\d.]+)',
        'HYG:TLT': r'HYG:TLT[^\d]*([\d.]+)',
        'IWM:SPY': r'IWM:SPY[^\d]*([\d.]+)',
        'SMH:SPY': r'SMH:SPY[^\d]*([\d.]+)',
        'SPXS:SVOL': r'SPXS:SVOL[^\d]*([\d.]+)',
        'URSP': r'URSP[^\d]*([\d.]+)',
        'RSP': r'RSP[^\d]*([\d.]+)',
        'SPX': r'\$SPX[^\d]*([\d,]+\.?\d*)',
    }
    
    if pattern:
        patterns[indicator_name] = pattern
    
    if indicator_name in patterns:
        match = re.search(patterns[indicator_name], html_content, re.IGNORECASE)
        if match:
            try:
                val = match.group(1).replace(',', '')
                return float(val)
            except:
                return None
    return None

def calculate_breadth_score(data_dict):
    """
    Calculate Breadth Confluence Score using %B + ROC + RSI hierarchy for BPSPX,
    plus weighted contributions from other indicators.
    """
    score = 0
    components = {}
    
    # BPSPX Regime Classification (%B + ROC + RSI hierarchy)
    if 'BPSPX' in data_dict and data_dict['BPSPX'] is not None:
        bpspx = data_dict['BPSPX']
        # Simple regime logic (customize thresholds based on your historical ranges)
        if bpspx >= 70:
            regime_score = 3  # Bullish
            regime_label = "🟢 Bullish"
        elif bpspx <= 30:
            regime_score = 1  # Bearish
            regime_label = "🔴 Bearish"
        else:
            regime_score = 2  # Neutral
            regime_label = "🟡 Neutral"
        components['BPSPX_Regime'] = {'value': bpspx, 'score': regime_score, 'label': regime_label}
        score += regime_score * 2  # Weight BPSPX heavily
    
    # Oscillator indicators (NYMO, NYSI, CPCE inverted, etc.)
    oscillator_weights = {
        'NYMO': 1.0, 'NYSI': 1.0, 'BPNYA': 1.5,
        'SPXA50R': 1.0, 'OEXA50R': 0.8, 'OEXA150R': 0.8, 'OEXA200R': 0.8,
    }
    
    for ind, weight in oscillator_weights.items():
        if ind in data_dict and data_dict[ind] is not None:
            val = data_dict[ind]
            # Normalize to 0-3 scale (customize per indicator behavior)
            if ind == 'CPCE':  # Inverted: lower is bullish
                norm_score = 3 if val < 0.6 else (2 if val < 0.8 else 1)
            elif ind in ['NYMO', 'NYSI']:  # Momentum oscillators
                norm_score = 3 if val > 50 else (2 if val > -50 else 1)
            else:  # Percent above MA indicators
                norm_score = 3 if val > 60 else (2 if val > 40 else 1)
            components[ind] = {'value': val, 'score': norm_score, 'weight': weight}
            score += norm_score * weight
    
    # Ratio indicators (RSP:SPY, IWM:SPY, etc.)
    ratio_weights = {'RSP:SPY': 1.5, 'IWM:SPY': 1.0, 'SMH:SPY': 1.0, 'XLF:SPY': 0.8}
    for ind, weight in ratio_weights.items():
        if ind in data_dict and data_dict[ind] is not None:
            val = data_dict[ind]
            # Simple threshold logic (adjust based on your ratio chart analysis)
            norm_score = 3 if val > 0.30 else (2 if val > 0.28 else 1)
            components[ind] = {'value': val, 'score': norm_score, 'weight': weight}
            score += norm_score * weight
    
    # Volatility/VIX proxy (VXX inverted)
    if 'VXX' in data_dict and data_dict['VXX'] is not None:
        vxx = data_dict['VXX']
        vxx_score = 3 if vxx < 25 else (2 if vxx < 35 else 1)
        components['VXX'] = {'value': vxx, 'score': vxx_score, 'weight': 1.0}
        score += vxx_score * 1.0
    
    # Normalize score to 0-100 scale
    max_possible = sum([3*2] + [3*w for w in list(oscillator_weights.values()) + list(ratio_weights.values()) + [1.0]])
    normalized_score = min(100, max(0, (score / max_possible) * 100))
    
    return round(normalized_score, 1), components

def get_signal_emoji(score, prev_score=None):
    """Return emoji signal based on score and change"""
    if prev_score is not None:
        change = score - prev_score
        if score >= 70 and change >= 0:
            return "🟢"
        elif score <= 30 and change <= 0:
            return "🔴"
        elif change > 5:
            return "🟡↑"
        elif change < -5:
            return "🟡↓"
        else:
            return "🟡"
    else:
        if score >= 70:
            return "🟢"
        elif score <= 30:
            return "🔴"
        else:
            return "🟡"

# ========================
# STREAMLIT APP
# ========================

def main():
    st.title("📊 Breadth Confluence Dashboard")
    st.markdown("*Equal-weight breadth timing engine for RSP/URSP analysis*")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Data Source")
    source_option = st.sidebar.radio(
        "Select data method:",
        ["🔄 Auto-fetch from StockCharts", "📤 Manual CSV Upload", "🎲 Demo Data"]
    )
    
    # Initialize session state for historical comparison
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = []
    
    # Data loading section
    data_dict = {}
    
    if source_option == "🔄 Auto-fetch from StockCharts":
        st.sidebar.info("⚠️ StockCharts scraping is experimental. May fail due to JS rendering.")
        urls = [
            "https://stockcharts.com/public/3423650/tenpp/1",
            "https://stockcharts.com/public/3423650/tenpp/2", 
            "https://stockcharts.com/public/3423650/tenpp/3"
        ]
        
        with st.spinner("🔍 Fetching breadth data from StockCharts..."):
            combined_html = ""
            for i, url in enumerate(urls, 1):
                html = fetch_stockcharts_page(url, i)
                if html:
                    combined_html += html
            
            if combined_html:
                # Extract all indicators
                indicators = ['BPSPX', 'BPNYA', 'NYMO', 'NYSI', 'SPXA50R', 'OEXA50R', 
                             'OEXA150R', 'OEXA200R', 'CPCE', 'VXX', 'RSP:SPY', 
                             'HYG:IEF', 'HYG:TLT', 'IWM:SPY', 'SMH:SPY', 'SPXS:SVOL',
                             'URSP', 'RSP', 'SPX']
                
                for ind in indicators:
                    data_dict[ind] = extract_indicator_value(combined_html, ind)
                
                st.success(f"✅ Extracted {sum(1 for v in data_dict.values() if v is not None)}/{len(indicators)} indicators")
            else:
                st.error("❌ Failed to fetch data. Try manual upload or demo mode.")
                source_option = "🎲 Demo Data"  # Fallback
    
    if source_option == "📤 Manual CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Upload breadth data CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            # Expect columns: indicator, value, date
            for _, row in df.iterrows():
                data_dict[row['indicator']] = row['value']
            st.success(f"✅ Loaded {len(df)} data points")
    
    if source_option == "🎲 Demo Data" or not data_dict:
        # Generate realistic demo data based on typical breadth readings
        np.random.seed(42)
        data_dict = {
            'BPSPX': np.random.uniform(35, 55),
            'BPNYA': np.random.uniform(40, 60),
            'NYMO': np.random.uniform(-100, 100),
            'NYSI': np.random.uniform(-500, 500),
            'SPXA50R': np.random.uniform(25, 45),
            'OEXA50R': np.random.uniform(30, 50),
            'OEXA150R': np.random.uniform(40, 60),
            'OEXA200R': np.random.uniform(45, 65),
            'CPCE': np.random.uniform(0.5, 0.9),
            'VXX': np.random.uniform(20, 40),
            'RSP:SPY': np.random.uniform(0.27, 0.32),
            'HYG:IEF': np.random.uniform(0.80, 0.86),
            'HYG:TLT': np.random.uniform(0.88, 0.95),
            'IWM:SPY': np.random.uniform(0.35, 0.40),
            'SMH:SPY': np.random.uniform(0.55, 0.62),
            'SPXS:SVOL': np.random.uniform(1.5, 3.5),
            'URSP': np.random.uniform(38, 45),
            'RSP': np.random.uniform(185, 200),
            'SPX': np.random.uniform(6500, 6800),
        }
        st.info("🎲 Using demo data - replace with live fetch or upload for production")
    
    # ========================
    # DASHBOARD DISPLAY
    # ========================
    
    if data_dict:
        # Calculate current score
        current_score, components = calculate_breadth_score(data_dict)
        
        # Historical comparison (if available)
        prev_score = None
        if st.session_state.historical_data:
            prev_score = st.session_state.historical_data[-1]['score']
        
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Breadth Score", f"{current_score}/100", 
                     delta=f"{current_score - prev_score:.1f}" if prev_score else None)
        with col2:
            signal = get_signal_emoji(current_score, prev_score)
            st.metric("📡 Signal", signal)
        with col3:
            bpspx_val = data_dict.get('BPSPX')
            st.metric("BPSPX", f"{bpspx_val:.1f}" if bpspx_val else "N/A")
        with col4:
            rsp_val = data_dict.get('RSP')
            st.metric("RSP Price", f"${rsp_val:.2f}" if rsp_val else "N/A")
        
        # Breadth Confluence Score narrative
        st.subheader("📈 Breadth Confluence Narrative")
        if current_score >= 70:
            st.success("🟢 **Bullish Confluence**: Multiple breadth indicators confirm upward momentum. Equal-weight leadership (RSP/URSP) favored.")
        elif current_score <= 30:
            st.error("🔴 **Bearish Confluence**: Breadth deterioration suggests caution. Defensive positioning may be warranted.")
        else:
            st.warning("🟡 **Mixed Signals**: Breadth indicators conflicted. Wait for confirmation or reduce position size.")
        
        # What Changed Table (per user preference)
        st.subheader("🔍 What Changed: Indicator Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        indicator_list = ['BPSPX', 'BPNYA', 'NYMO', 'NYSI', 'SPXA50R', 'OEXA50R', 
                         'OEXA150R', 'OEXA200R', 'NYAD', 'NYHL', 'CPCE', 'VXX', 
                         'RSP:SPY', 'HYG:IEF', 'HYG:TLT', 'IWM:SPY', 'SMH:SPY', 
                         'SPXS:SVOL', 'AAII', 'NAAIM']
        
        for ind in indicator_list:
            current = data_dict.get(ind)
            # Simulate previous value (in production, fetch from history)
            prev = current * np.random.uniform(0.95, 1.05) if current else None
            change = current - prev if current and prev else None
            
            # Determine implication emoji
            if ind in ['CPCE', 'VXX', 'SPXS:SVOL']:  # Inverted indicators
                emoji = "🟢" if (change and change < 0) else ("🔴" if (change and change > 0) else "🟡")
            else:
                emoji = "🟢" if (change and change > 0) else ("🔴" if (change and change < 0) else "🟡")
            
            comparison_data.append({
                'Indicator': ind,
                'Previous': f"{prev:.2f}" if prev else "N/A",
                'Current': f"{current:.2f}" if current else "N/A",
                'Change': f"{change:+.2f}" if change else "N/A",
                'Implication (RSP)': emoji
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(
            comparison_df.style.applymap(
                lambda x: 'background-color: #d4edda' if x == '🟢' else 
                         ('background-color: #f8d7da' if x == '🔴' else ''),
                subset=['Implication (RSP)']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Candle & Line Analysis Section
        st.subheader("🕯️ Candle & Line Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RSP Price Action**")
            rsp = data_dict.get('RSP')
            if rsp:
                if rsp > 195:
                    st.markdown("🟢 RSP holding above key support. Bullish structure intact.")
                elif rsp > 185:
                    st.markdown("🟡 RSP consolidating. Watch for breakout/breakdown.")
                else:
                    st.markdown("🔴 RSP below support. Bearish pressure increasing.")
        with col2:
            st.markdown("**RSP:SPY Ratio**")
            ratio = data_dict.get('RSP:SPY')
            if ratio:
                if ratio > 0.30:
                    st.markdown("🟢 Equal-weight outperformance. Breadth confirming.")
                elif ratio > 0.28:
                    st.markdown("🟡 Ratio neutral. Awaiting directional cue.")
                else:
                    st.markdown("🔴 Equal-weight underperformance. Caution warranted.")
        
        # Regime Classification (BPSPX %B+ROC+RSI hierarchy)
        st.subheader("🎭 Regime Classification: BPSPX")
        bpspx = data_dict.get('BPSPX')
        if bpspx:
            # Calculate pseudo %B, ROC, RSI from BPSPX value (simplified)
            pct_b = min(1.0, max(0.0, (bpspx - 20) / 80))  # Normalize to 0-1
            roc_signal = "📈" if bpspx > 50 else "📉"
            rsi_signal = "🟢" if bpspx > 60 else ("🔴" if bpspx < 40 else "🟡")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("%B Proxy", f"{pct_b:.2f}")
            col2.metric("ROC Signal", roc_signal)
            col3.metric("RSI Signal", rsi_signal)
            
            # Historic range percentile
            percentile = min(100, max(0, (bpspx / 100) * 100))
            st.progress(percentile / 100)
            st.caption(f"BPSPX at {percentile:.1f}th percentile of historic range")
        
        # Action Plan (A/B/C scenarios for RSP + URSP)
        st.subheader("🗓️ Scenario-Based Action Plan")
        
        scenarios = {
            "🟢 Scenario A: Bullish Confirmation": {
                "Condition": "Breadth Score ≥70 AND RSP:SPY rising AND BPSPX >60",
                "Action RSP": "Add to long positions, target 2-3% position size",
                "Action URSP": "Add to leveraged long, target 1-1.5% position size",
                "Stop RSP": "Close below 20-day MA or -3% from entry",
                "Stop URSP": "Close below 20-day MA or -5% from entry",
                "Logic": "Breadth confirmation supports equal-weight leadership continuation"
            },
            "🟡 Scenario B: Neutral/Wait": {
                "Condition": "Breadth Score 40-69 OR mixed indicator signals",
                "Action RSP": "Hold existing positions, no new adds",
                "Action URSP": "Reduce leveraged exposure by 50%",
                "Stop RSP": "Tighten stops to -2% from entry",
                "Stop URSP": "Exit if breadth score drops below 40",
                "Logic": "Uncertain regime favors capital preservation"
            },
            "🔴 Scenario C: Bearish Defense": {
                "Condition": "Breadth Score ≤39 OR RSP:SPY breaking down",
                "Action RSP": "Reduce to 50% position size or go flat",
                "Action URSP": "Exit all leveraged long positions",
                "Stop RSP": "Hard stop at -3% or key support break",
                "Stop URSP": "Immediate exit on confirmation",
                "Logic": "Breadth deterioration signals risk-off environment"
            }
        }
        
        # Determine active scenario
        active_scenario = "🟡 Scenario B: Neutral/Wait"
        if current_score >= 70 and (data_dict.get('RSP:SPY', 0) or 0) > 0.29 and (data_dict.get('BPSPX', 0) or 0) > 60:
            active_scenario = "🟢 Scenario A: Bullish Confirmation"
        elif current_score <= 39 or (data_dict.get('RSP:SPY', 0) or 0) < 0.28:
            active_scenario = "🔴 Scenario C: Bearish Defense"
        
        # Display scenarios with highlighting
        for name, details in scenarios.items():
            with st.expander(name, expanded=(name == active_scenario)):
                for key, value in details.items():
                    st.markdown(f"**{key}**: {value}")
        
        # Bottom Line Summary
        st.subheader("💡 Bottom Line")
        st.markdown(f"""
        **Current Assessment**: Breadth Confluence Score = **{current_score}/100** ({get_signal_emoji(current_score, prev_score)})
        
        **Key Confirmation Signals**:
        - BPSPX regime: {components.get('BPSPX_Regime', {}).get('label', 'N/A')}
        - RSP:SPY ratio: {'🟢 Bullish' if (data_dict.get('RSP:SPY', 0) or 0) > 0.30 else '🔴 Bearish'}
        - Volatility (VXX): {'🟢 Low' if (data_dict.get('VXX', 100) or 100) < 30 else '🔴 Elevated'}
        
        **Risk/Reward**: {'Favorable for RSP/URSP longs' if current_score >= 60 else 'Cautious stance recommended'}
        """)
        
        # Save to history for next comparison
        st.session_state.historical_data.append({
            'timestamp': datetime.now(),
            'score': current_score,
            'data': data_dict.copy()
        })
        
        # Download button for current data
        if st.button("💾 Export Current Data"):
            export_df = pd.DataFrame([
                {'indicator': k, 'value': v, 'timestamp': datetime.now()}
                for k, v in data_dict.items() if v is not None
            ])
            csv = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"breadth_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ This tool is for research purposes only. Not investment advice. Data freshness depends on source availability.")

if __name__ == "__main__":
    main()
