import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.info("AUDPC summarizes the entire season's mite pressure into a single value.")

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
    
    model_formula = 'AUDPC_Value ~ C(Crop) + C(Field_Type) + C(Crop):C(Field_Type)'
    
    try:
        model = ols(model_formula, data=audpc_results).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        st.write("**ANOVA Table Results:**")
        st.dataframe(anova_table.style.format(precision=4))

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
    audpc_results['Treatment'] = audpc_results['Crop'] + " - " + audpc_results['Field_Type']
    
    if st.button("Run Tukey's HSD Test"):
        tukey = pairwise_tukeyhsd(endog=audpc_results['AUDPC_Value'],
                                groups=audpc_results['Treatment'],
                                 alpha=0.05)
        st.write("**Pairwise Comparisons:**")
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        st.dataframe(tukey_df)

    st.download_button(
        label="Download AUDPC Data CSV",
        data=audpc_results.to_csv(index=False).encode('utf-8'),
        file_name='phase2_audpc_results.csv',
        mime='text/csv',
    )

    # --- PHASE 3: TEMPORAL PATTERN ANALYSES ---
    st.divider()
    st.title("🎯 Statistical Analysis: Phase 3")
    st.markdown("### Temporal Pattern Analyses")

    # Step 3.1: Mixed Model Repeated Measures
    st.subheader("📋 Step 3.1: Mixed Model (Repeated Measures)")
    try:
        mixed_formula = 'Mite_Count ~ C(Crop) * C(Field_Type) * Week'
        random_grp = 'Year'
        if 'Replicate' in df_final.columns:
            df_final['Group_ID'] = df_final['Year'].astype(str) + "_" + df_final['Replicate'].astype(str)
            random_grp = 'Group_ID'

        model_mixed = sm.MixedLM.from_formula(mixed_formula, groups=df_final[random_grp], data=df_final)
        mixed_results = model_mixed.fit()
        st.text(mixed_results.summary())
    except Exception as e:
        st.error(f"Mixed Model Error: {e}")

    # Step 3.2: Peak Week Identification
    st.subheader("📋 Step 3.2: Peak Week Identification")
    peak_analysis = df_final.groupby(['Crop', 'Field_Type', 'Week'])['Mite_Count'].mean().reset_index()
    idx = peak_analysis.groupby(['Crop', 'Field_Type'])['Mite_Count'].idxmax()
    peak_results = peak_analysis.loc[idx].rename(columns={'Week': 'Peak_Week', 'Mite_Count': 'Peak_Density'})
    st.dataframe(peak_results.style.format({"Peak_Density": "{:.2f}"}))

    # Step 3.3: Economic Threshold Analysis
    st.subheader("📋 Step 3.3: Economic Threshold Analysis")
    threshold = st.number_input("Set Economic Threshold (Mites/Plant)", value=2.0)
    
    threshold_data = df_final.groupby(['Year', 'Crop', 'Field_Type']).apply(
        lambda x: x[x['Mite_Count'] >= threshold]['Week'].min() if not x[x['Mite_Count'] >= threshold].empty else np.nan
    ).reset_index()
    threshold_data.columns = ['Year', 'Crop', 'Field_Type', 'Threshold_Week']
    
    st.write("Average Week Reaching Threshold:")
    st.dataframe(threshold_data.groupby(['Crop', 'Field_Type'])['Threshold_Week'].mean())

    # Visualization
    st.subheader("📈 Temporal Dynamics Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_final, x='Week', y='Mite_Count', hue='Crop', style='Field_Type', marker='o', ax=ax)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    st.pyplot(fig)

    # --- PHASE 4: CROP-SPECIFIC ANALYSES ---
    st.divider()
    st.title("🎯 Statistical Analysis: Phase 4")
    st.markdown("### Crop-Specific Impact Analysis")

    crops = df_final['Crop'].unique()
    
    for selected_crop in crops:
        st.subheader(f"🌱 Analysis for {selected_crop}")
        
        # Subset Data
        crop_data = audpc_results[audpc_results['Crop'] == selected_crop]
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**One-Way ANOVA (Field Type Effect)**")
            try:
                formula_crop = 'AUDPC_Value ~ C(Field_Type)'
                model_crop = ols(formula_crop, data=crop_data).fit()
                anova_crop = sm.stats.anova_lm(model_crop, typ=2)
                st.dataframe(anova_crop.style.format(precision=4))
            except Exception as e:
                st.error(f"Error in ANOVA: {e}")
                
        with col_b:
            st.markdown("**Efficiency of Organic Management**")
            means = crop_data.groupby('Field_Type')['AUDPC_Value'].mean()
            if 'Organic' in means and 'Non-organic' in means:
                reduction = ((means['Non-organic'] - means['Organic']) / means['Non-organic']) * 100
                st.metric(label=f"Mite Reduction in {selected_crop}", value=f"{reduction:.2f}%", help="Reduction in AUDPC compared to Non-organic")
            else:
                st.warning("Ensure 'Field_Type' contains both 'Organic' and 'Non-organic' labels for % calculation.")

        # Mixed Model for this specific crop
        with st.expander(f"View Temporal Model for {selected_crop}"):
            try:
                crop_temporal = df_final[df_final['Crop'] == selected_crop]
                formula_mixed_crop = 'Mite_Count ~ C(Field_Type) * Week'
                model_mixed_crop = sm.MixedLM.from_formula(formula_mixed_crop, groups=crop_temporal[random_grp], data=crop_temporal)
                results_mixed_crop = model_mixed_crop.fit()
                st.text(results_mixed_crop.summary())
            except Exception as e:
                st.error(f"Error in Mixed Model: {e}")

else:
    st.info("Please upload a CSV file to begin.")
