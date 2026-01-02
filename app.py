import streamlit as st
import pandas as pd
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Mite Analysis Tool", layout="wide")

def calculate_audpc(df):
    """Calculates Area Under Disease Progress Curve (AUDPC)"""
    # Sort by week to ensure correct calculation
    df = df.sort_values('week')
    weeks = df['week'].values
    counts = df['mite_count'].values
    
    audpc = 0
    for i in range(len(df) - 1):
        # Formula: ((y1 + y2) / 2) * (x2 - x1)
        mean_mite = (counts[i] + counts[i+1]) / 2
        interval = weeks[i+1] - weeks[i]
        audpc += mean_mite * interval
    return round(audpc, 2)

def analyze_data(df):
    """Processes the dataframe to generate statistics"""
    results = []
    # Group by Crop and Field Type
    grouped = df.groupby(['crop', 'field_type'])
    
    for (crop, field), group in grouped:
        audpc = calculate_audpc(group)
        peak_val = group['mite_count'].max()
        peak_week = group.loc[group['mite_count'] == peak_val, 'week'].iloc[0]
        
        # Find first week reaching threshold of 2
        threshold_df = group[group['mite_count'] >= 2]
        threshold_week = threshold_df['week'].min() if not threshold_df.empty else "Not Reached"
        
        results.append({
            "Treatment": f"{crop} ({field})",
            "Crop": crop,
            "Field Type": field,
            "AUDPC": audpc,
            "Peak Density": peak_val,
            "Peak Week": peak_week,
            "Threshold Week": threshold_week
        })
    return pd.DataFrame(results)

# --- USER INTERFACE ---
st.title("🔬 Mite Population Analysis System")
st.markdown("Automated analysis for Okra and Brinjal mite dynamics.")

# Create Tabs
tab_upload, tab_summary, tab_graphs, tab_tables = st.tabs(["📁 Upload Data", "📊 Summary", "📈 Graphs", "📋 Data Table"])

with tab_upload:
    st.header("1. Upload CSV Data")
    uploaded_file = st.file_uploader("Choose your mite data CSV file", type="csv")
    
    with st.expander("View Required CSV Format"):
        st.write("Your CSV must have these columns (any capitalization):")
        st.code("week, crop, field_type, mite_count")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # --- CRITICAL FIX FOR KEYERROR ---
            # This makes all column names lowercase and removes hidden spaces
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Verify required columns exist
            required = ['week', 'crop', 'field_type', 'mite_count']
            if all(col in df.columns for col in required):
                st.session_state['data'] = df
                st.success("File uploaded and verified! Proceed to other tabs.")
            else:
                st.error(f"Missing columns! Your file must contain: {', '.join(required)}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Only show results if data is uploaded
if 'data' in st.session_state:
    data = st.session_state['data']
    analysis_results = analyze_data(data)

    with tab_summary:
        st.header("Statistical Key Metrics")
        cols = st.columns(2)
        for i, row in analysis_results.iterrows():
            with cols[i % 2]:
                st.subheader(row['Treatment'])
                st.metric("AUDPC Value", row['AUDPC'])
                st.write(f"**Peak Density:** {row['Peak Density']} mites/plant (Week {row['Peak Week']})")
                st.write(f"**Threshold (2/plant):** Week {row['Threshold Week']}")
                st.divider()

    with tab_graphs:
        st.header("Population Trends Over Time")
        # Prepare data for plotting
        chart_data = data.pivot_table(
            index='week', 
            columns=['crop', 'field_type'], 
            values='mite_count'
        ).reset_index()
        
        # Flatten the multi-level columns created by pivot
        chart_data.columns = [f"{c[0]}-{c[1]}" if c[1] else c[0] for c in chart_data.columns]
        
        st.line_chart(chart_data.set_index('week'))
        st.caption("X-Axis: Week Number | Y-Axis: Mite Count per Plant")

    with tab_tables:
        st.header("Detailed Analysis Table")
        st.dataframe(analysis_results, use_container_width=True)
        
        # CSV Export
        csv_download = analysis_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_download,
            file_name="mite_analysis_report.csv",
            mime="text/csv",
        )
else:
    with tab_summary: st.info("Please upload a CSV file in the Upload tab.")
    with tab_graphs: st.info("Waiting for data...")
    with tab_tables: st.info("Waiting for data...")
