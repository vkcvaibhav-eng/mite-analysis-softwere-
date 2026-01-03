import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set Page Config
st.set_page_config(page_title="Mite Analysis Tool", layout="wide")

# --- PHASE 1: DATA PREPARATION & EXPLORATION ---
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
    
    group_cols = ['Crop', 'Field_Type']
    stats_df = df_final.groupby(group_cols)['Mite_Count'].agg([
        ('Mean', 'mean'),
        ('SD', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Sample_Size', 'count')
    ]).reset_index()

    stats_df['Standard_Error'] = stats_df['SD'] / np.sqrt(stats_df['Sample_Size'])
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

    # --- PHASE 2: PRIMARY STATISTICAL ANALYSES ---
    st.divider()
    st.title("🎯 Statistical Analysis: Phase 2")
    st.markdown("### AUDPC Calculation & Primary ANOVA")

    # --- STEP 2.1: AUDPC CALCULATION ---
    st.subheader("📋 Step 2.1: AUDPC Calculation")
    st.info("AUDPC summarizes the entire season's mite pressure into a single value for each plot/replicate.")

    def calculate_audpc(group):
        group = group.sort_values('Week')
        y = group['Mite_Count'].values
        t = group['Week'].values
        # AUDPC Formula: Σ [(Yi + Yi+1)/2] × (ti+1 - ti)
        audpc_val = np.sum((y[:-1] + y[1:]) / 2 * np.diff(t))
        return audpc_val

    # Grouping logic for AUDPC
    grouping_cols = ['Year', 'Crop', 'Field_Type']
    if 'Replicate' in df_final.columns:
        grouping_cols.append('Replicate')

    audpc_results = df_final.groupby(grouping_cols).apply(calculate_audpc).reset_index()
    audpc_results.columns = grouping_cols + ['AUDPC_Value']

    st.write("Calculated AUDPC Values (First 10 rows):")
    st.dataframe(audpc_results.head(10).style.format({"AUDPC_Value": "{:.2f}"}))

    # --- STEP 2.2: TWO-WAY ANOVA ON AUDPC ---
    st.subheader("📋 Step 2.2: Two-Way ANOVA (Main Test)")
    
    # Building the Model: AUDPC ~ Crop + Field_Type + Crop:Field_Type
    model_formula = 'AUDPC_Value ~ C(Crop) + C(Field_Type) + C(Crop):C(Field_Type)'
    
    try:
        model = ols(model_formula, data=audpc_results).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        st.write("**ANOVA Table Results:**")
        st.dataframe(anova_table.style.format(precision=4))

        # Interpretation logic
        sig_threshold = 0.05
        st.markdown("#### 🔍 Key Findings:")
        for factor in anova_table.index[:-1]:
            p_val = anova_table.loc[factor, 'PR(>F)']
            if p_val < sig_threshold:
                st.success(f"✅ **{factor}** is statistically significant (p = {p_val:.4f})")
            else:
                st.warning(f"⚪ **{factor}** is NOT statistically significant (p = {p_val:.4f})")

    except Exception as e:
        st.error(f"Error running ANOVA: {e}")

    # --- STEP 2.3: POST-HOC TESTS (TUKEY HSD) ---
    st.subheader("📋 Step 2.3: Tukey's Post-Hoc Test")
    
    # Create a 'Treatment' column for pairwise comparison
    audpc_results['Treatment'] = audpc_results['Crop'] + " - " + audpc_results['Field_Type']
    
    if st.button("Run Tukey's HSD Test"):
        tukey = pairwise_tukeyhsd(endog=audpc_results['AUDPC_Value'],
                                  groups=audpc_results['Treatment'],
                                  alpha=0.05)
        
        st.write("**Pairwise Comparisons:**")
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        st.dataframe(tukey_df)
        
        sig_diffs = tukey_df[tukey_df['reject'] == True]
        if not sig_diffs.empty:
            st.success(f"Significant differences found in {len(sig_diffs)} treatment pairs.")
        else:
            st.info("No significant differences found between treatment groups.")

    # --- DOWNLOAD RESULTS ---
    st.subheader("💾 Export Phase 2 Results")
    csv_audpc = audpc_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download AUDPC Data CSV",
        data=csv_audpc,
        file_name='phase2_audpc_results.csv',
        mime='text/csv',
    )

else:
    st.info("Please upload a CSV file to begin Phase 1.")
    st.warning("Phase 2 will become available once data is uploaded and processed.")
