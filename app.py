import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io

# Set Page Config
st.set_page_config(page_title="Mite Analysis Tool", layout="wide")

# --- PDF REPORTING CLASS ---
class ResearchPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Mite Population Research Analysis Report', 0, 1, 'C')
        self.ln(5)

    def create_table(self, df, title):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_font('Arial', '', 8)
        col_width = self.epw / len(df.columns)
        for col in df.columns:
            self.cell(col_width, 7, str(col), border=1)
        self.ln()
        for row in df.values:
            for datum in row:
                self.cell(col_width, 6, str(datum), border=1)
            self.ln()
        self.ln(5)

# --- PHASE 1: DATA PREPARATION & EXPLORATION ---
st.title("🎯 Statistical Analysis Roadmap")
st.markdown("### Phase 1: Data Preparation & Exploration")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    st.subheader("📋 Step 1.1: Data Organization & Factor Mapping")
    
    col1, col2 = st.columns(2)
    with col1:
        year_col = st.selectbox("Select Year Column", df_raw.columns, index=0 if 'Year' in df_raw.columns else 0)
        week_col = st.selectbox("Select Week Column", df_raw.columns, index=1 if 'Week' in df_raw.columns else 0)
        crop_col = st.selectbox("Select Crop (Factor 1)", df_raw.columns, index=2 if 'Crop' in df_raw.columns else 0)
    
    with col2:
        mgmt_col = st.selectbox("Select Field Management (Factor 2)", df_raw.columns, index=3 if 'Management' in df_raw.columns else 0)
        mite_col = st.selectbox("Select Mite Count (Dependent Variable)", df_raw.columns, index=4 if 'Mite_Count' in df_raw.columns else 0)
        rep_col = st.selectbox("Select Replicate Column (if any)", ["None"] + list(df_raw.columns))

    df = df_raw.copy()
    mapping = {year_col: 'Year', week_col: 'Week', crop_col: 'Crop', mgmt_col: 'Field_Type', mite_col: 'Mite_Count'}
    df = df.rename(columns=mapping)
    
    selected_cols = ['Year', 'Week', 'Crop', 'Field_Type', 'Mite_Count']
    if rep_col != "None":
        df = df.rename(columns={rep_col: 'Replicate'})
        selected_cols.append('Replicate')
    
    df_final = df[selected_cols]

    # Descriptive Stats
    st.subheader("📊 Step 1.2: Descriptive Statistics")
    group_cols = ['Crop', 'Field_Type']
    stats_df = df_final.groupby(group_cols)['Mite_Count'].agg([
        ('Sample_Size', 'count'), ('Mean', 'mean'), ('SD', 'std'), ('Min', 'min'), ('Max', 'max')
    ]).reset_index()
    stats_df['Standard_Error'] = stats_df['SD'] / np.sqrt(stats_df['Sample_Size'])
    st.dataframe(stats_df.style.format(precision=3))

    # --- PHASE 2: PRIMARY STATISTICAL ANALYSES ---
    st.divider()
    st.title("🎯 Statistical Analysis: Phase 2")
    
    def calculate_audpc(group):
        group = group.sort_values('Week')
        y = group['Mite_Count'].values
        t = group['Week'].values
        return np.sum((y[:-1] + y[1:]) / 2 * np.diff(t))

    grouping_cols = ['Year', 'Crop', 'Field_Type']
    if 'Replicate' in df_final.columns: grouping_cols.append('Replicate')
    audpc_results = df_final.groupby(grouping_cols).apply(calculate_audpc).reset_index()
    audpc_results.columns = grouping_cols + ['AUDPC_Value']

    st.subheader("📋 Step 2.2: Two-Way ANOVA (AUDPC)")
    model_formula = 'AUDPC_Value ~ C(Crop) + C(Field_Type) + C(Crop):C(Field_Type)'
    model = ols(model_formula, data=audpc_results).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.dataframe(anova_table.style.format(precision=4))

    # Tukey HSD
    audpc_results['Treatment'] = audpc_results['Crop'] + " - " + audpc_results['Field_Type']
    tukey = pairwise_tukeyhsd(endog=audpc_results['AUDPC_Value'], groups=audpc_results['Treatment'], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    # --- PHASE 3: TEMPORAL PATTERN ANALYSES ---
    st.divider()
    st.title("🎯 Statistical Analysis: Phase 3")
    peak_analysis = df_final.groupby(['Crop', 'Field_Type', 'Week'])['Mite_Count'].mean().reset_index()
    peak_idx = peak_analysis.groupby(['Crop', 'Field_Type'])['Mite_Count'].idxmax()
    peak_results = peak_analysis.loc[peak_idx].rename(columns={'Week': 'Peak_Week', 'Mite_Count': 'Peak_Density'})

    threshold = st.number_input("Set Economic Threshold (Mites/Plant)", value=2.0)
    threshold_data = df_final.groupby(['Year', 'Crop', 'Field_Type']).apply(
        lambda x: x[x['Mite_Count'] >= threshold]['Week'].min() if not x[x['Mite_Count'] >= threshold].empty else np.nan
    ).reset_index()
    threshold_data.columns = ['Year', 'Crop', 'Field_Type', 'Threshold_Week']

    # --- PHASE 4 & 5: COMPARATIVE & EFFECT SIZES ---
    st.divider()
    st.title("🎯 Phases 4 & 5: Effect Sizes")
    ss_treatment = anova_table['sum_sq'].iloc[0:3].sum()
    ss_total = anova_table['sum_sq'].sum()
    eta_sq = ss_treatment / ss_total
    st.metric("Overall Eta-squared (η²)", f"{eta_sq:.3f}")

    # --- PHASE 7: PUBLICATION REPORT GENERATION ---
    st.divider()
    st.title("📊 PHASE 7: RESEARCH PAPER TABLES")
    
    # --- TABLE 1: Descriptive Stats Format ---
    t1_pub = stats_df.copy()
    t1_pub['Mean ± SE'] = t1_pub.apply(lambda x: f"{x['Mean']:.2f} ± {x['Standard_Error']:.2f}", axis=1)
    t1_pub['CV (%)'] = (t1_pub['SD'] / t1_pub['Mean'] * 100).round(2)
    t1_pub = t1_pub[['Crop', 'Field_Type', 'Mean ± SE', 'SD', 'Min', 'Max', 'CV (%)', 'Sample_Size']]
    st.subheader("TABLE 1: DESCRIPTIVE STATISTICS")
    st.dataframe(t1_pub)

    # --- TABLE 2: ANOVA Result Format ---
    t2_pub = anova_table.copy().reset_index()
    t2_pub.columns = ['Source', 'Sum_Sq', 'df', 'F', 'p-value']
    def sig_code(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"
    t2_pub['Significance'] = t2_pub['p-value'].apply(sig_code)
    st.subheader("TABLE 2: ANOVA RESULTS (AUDPC)")
    st.dataframe(t2_pub)

    # --- TABLE 5: Peak Parameters Format ---
    avg_thresh_week = threshold_data.groupby(['Crop', 'Field_Type'])['Threshold_Week'].mean().reset_index()
    t5_pub = pd.merge(peak_results, avg_thresh_week, on=['Crop', 'Field_Type'])
    st.subheader("TABLE 5: PEAK POPULATION PARAMETERS")
    st.dataframe(t5_pub)

    # --- EXPORT SECTION ---
    st.markdown("### 📥 Download Results")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.download_button("Download Table 1 (CSV)", t1_pub.to_csv(index=False), "Table1_Descriptive.csv", "text/csv")
        st.download_button("Download Table 2 (CSV)", t2_pub.to_csv(index=False), "Table2_ANOVA.csv", "text/csv")

    with col_dl2:
        if st.button("Generate Professional PDF Report"):
            pdf = ResearchPDF()
            pdf.add_page()
            pdf.create_table(t1_pub, "Table 1: Descriptive Statistics of Mite Populations")
            pdf.create_table(t2_pub, "Table 2: ANOVA Results for AUDPC")
            pdf.create_table(t5_pub, "Table 5: Peak Population and Threshold Parameters")
            
            pdf_bytes = pdf.output()
            st.download_button("📥 Download PDF Report", data=bytes(pdf_bytes), file_name="Mite_Analysis_Report.pdf", mime="application/pdf")

else:
    st.info("Please upload a CSV file to begin the analysis.")
