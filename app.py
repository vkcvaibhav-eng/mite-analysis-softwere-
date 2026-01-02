import streamlit as st
import pandas as pd
import numpy as np

# Set page title
st.set_page_config(page_title="Mite Population Analysis", layout="wide")

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
    results = []
    # Group by Crop and Field Type
    grouped = df.groupby(['crop', 'field_type'])
    
    for (crop, field), group in grouped:
        audpc = calculate_audpc(group)
        peak_val = group['mite_count'].max()
        peak_week = group.loc[group['mite_count'] == peak_val, 'week'].iloc[0]
        
        # Find first week reaching threshold of 2
        threshold_df = group[group['mite_count'] >= 2]
        threshold_week = threshold_df['week'].min() if not threshold_df.empty else "N/A"
        
        results.append({
            "Treatment": f"{crop}-{field}",
            "Crop": crop,
            "Field Type": field,
            "AUDPC": audpc,
            "Peak Density": peak_val,
            "Peak Week": peak_week,
            "Threshold Week": threshold_week
        })
    return pd.DataFrame(results)

# --- UI ---
st.title("🔬 Mite Population Statistical Analysis System")

# Sidebar/Tabs equivalent
tab_upload, tab_summary, tab_graphs, tab_tables = st.tabs(["Upload", "Summary", "Graphs", "Tables"])

with tab_upload:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.info("Required columns: week, crop, field_type, mite_count")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.success("File uploaded successfully! Move to the next tabs to see results.")

if 'data' in st.session_state:
    data = st.session_state['data']
    analysis_results = analyze_data(data)

    with tab_summary:
        st.header("Analysis Summary")
        cols = st.columns(2)
        for i, row in analysis_results.iterrows():
            with cols[i % 2]:
                st.metric(label=row['Treatment'], value=f"AUDPC: {row['AUDPC']}")
                st.write(f"**Peak:** {row['Peak Density']} (Week {row['Peak Week']})")
                st.write(f"**Threshold Week:** {row['Threshold Week']}")
                st.markdown("---")

    with tab_graphs:
        st.header("Population Dynamics")
        # Pivot data for charting
        chart_data = data.pivot_table(index='week', columns=['crop', 'field_type'], values='mite_count').reset_index()
        chart_data.columns = [f"{c[0]}-{c[1]}" if isinstance(c, tuple) and c[1] else c[0] for c in chart_data.columns]
        
        st.line_chart(chart_data.set_index('week'))

    with tab_tables:
        st.header("Data Table")
        st.dataframe(analysis_results)
        
        # Export option
        csv = analysis_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Analysis as CSV", data=csv, file_name="AUDPC_results.csv", mime="text/csv")
else:
    with tab_summary: st.warning("Please upload a file first.")
    with tab_graphs: st.warning("Please upload a file first.")
    with tab_tables: st.warning("Please upload a file first.")
