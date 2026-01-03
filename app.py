import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# Set Page Config
st.set_page_config(page_title="Mite Analysis - Phase 1", layout="wide")

st.title("🎯 Statistical Analysis: Phase 1")
st.markdown("### Data Preparation & Exploration")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    st.subheader("📋 Step 1.1: Data Organization & Factor Mapping")
    st.write("Current data columns:", list(df_raw.columns))
    
    # Dynamic Mapping of Columns
    col1, col2 = st.columns(2)
    with col1:
        year_col = st.selectbox("Select Year Column", df_raw.columns, index=0 if 'Year' in df_raw.columns else 0)
        week_col = st.selectbox("Select Week Column", df_raw.columns, index=1 if 'Week' in df_raw.columns else 0)
        crop_col = st.selectbox("Select Crop (Factor 1)", df_raw.columns, index=2 if 'Crop' in df_raw.columns else 0)
    
    with col2:
        mgmt_col = st.selectbox("Select Field Management (Factor 2)", df_raw.columns, index=3 if 'Management' in df_raw.columns else 0)
        mite_col = st.selectbox("Select Mite Count (Dependent Variable)", df_raw.columns, index=4 if 'Mite_Count' in df_raw.columns else 0)
        rep_col = st.selectbox("Select Replicate Column (if any)", ["None"] + list(df_raw.columns))

    # Rename and Prepare Data
    df = df_raw.copy()
    mapping = {year_col: 'Year', week_col: 'Week', crop_col: 'Crop', mgmt_col: 'Field_Type', mite_col: 'Mite_Count'}
    df = df.rename(columns=mapping)
    
    # Subset to necessary columns
    selected_cols = ['Year', 'Week', 'Crop', 'Field_Type', 'Mite_Count']
    if rep_col != "None":
        df = df.rename(columns={rep_col: 'Replicate'})
        selected_cols.append('Replicate')
    
    df_final = df[selected_cols]

    # Data Quality Check
    st.info("📊 Data Integrity Report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", df_final.shape[0])
    c2.metric("Missing Values", df_final.isnull().sum().sum())
    c3.metric("Unique Years", df_final['Year'].nunique())

    # Preview Data
    with st.expander("🔍 View Processed Data Preview"):
        st.dataframe(df_final.head(10))

    # Step 1.2: Descriptive Statistics
    st.subheader("📊 Step 1.2: Descriptive Statistics")
    
    # Grouping logic
    group_cols = ['Crop', 'Field_Type']
    
    # Calculate Stats
    stats_df = df_final.groupby(group_cols)['Mite_Count'].agg([
        ('Mean', 'mean'),
        ('SD', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Sample_Size', 'count')
    ]).reset_index()

    # Calculate Standard Error (SE = SD / sqrt(n))
    stats_df['Standard_Error'] = stats_df['SD'] / np.sqrt(stats_df['Sample_Size'])

    # Reorder columns for professional look
    stats_df = stats_df[['Crop', 'Field_Type', 'Sample_Size', 'Mean', 'Standard_Error', 'SD', 'Min', 'Max']]

    st.write("Summary Statistics by Treatment Combination:")
    st.dataframe(stats_df.style.format(precision=3))

    # Outlier Detection (Z-Score)
    st.subheader("🚨 Outlier Analysis")
    df_final['Z_Score'] = stats.zscore(df_final['Mite_Count'])
    outliers = df_final[np.abs(df_final['Z_Score']) > 3]
    
    if not outliers.empty:
        st.warning(f"Found {len(outliers)} potential outliers (Z-score > 3).")
        st.dataframe(outliers)
    else:
        st.success("No extreme outliers detected (Z-score > 3).")

    # CSV Export of Stats
    st.download_button(
        label="Download Descriptive Statistics CSV",
        data=stats_df.to_csv(index=False).encode('utf-8'),
        file_name='phase1_descriptive_stats.csv',
        mime='text/csv',
    )

else:
    st.info("Please upload a CSV file to begin Phase 1.")
