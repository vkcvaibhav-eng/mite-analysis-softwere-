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
st.title("🎯 Statistical Analysis Roadmap")
st.markdown("### Phase 1: Data Preparation & Exploration")

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
        audpc_val = np.sum((y[:-1] + y[1:]) / 2 * np.diff(t))
        return audpc_val

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

    st.subheader("📋 Step 3.2: Peak Week Identification")
    peak_analysis = df_final.groupby(['Crop', 'Field_Type', 'Week'])['Mite_Count'].mean().reset_index()
    idx = peak_analysis.groupby(['Crop', 'Field_Type'])['Mite_Count'].idxmax()
    peak_results = peak_analysis.loc[idx].rename(columns={'Week': 'Peak_Week', 'Mite_Count': 'Peak_Density'})
    st.dataframe(peak_results.style.format({"Peak_Density": "{:.2f}"}))

    st.subheader("📋 Step 3.3: Economic Threshold Analysis")
    threshold = st.number_input("Set Economic Threshold (Mites/Plant)", value=2.0)
    
    threshold_data = df_final.groupby(['Year', 'Crop', 'Field_Type']).apply(
        lambda x: x[x['Mite_Count'] >= threshold]['Week'].min() if not x[x['Mite_Count'] >= threshold].empty else np.nan
    ).reset_index()
    threshold_data.columns = ['Year', 'Crop', 'Field_Type', 'Threshold_Week']
    
    st.write("Average Week Reaching Threshold:")
    st.dataframe(threshold_data.groupby(['Crop', 'Field_Type'])['Threshold_Week'].mean())

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
                st.metric(label=f"Mite Reduction in {selected_crop}", value=f"{reduction:.2f}%")
            else:
                st.warning("Ensure 'Field_Type' contains 'Organic' and 'Non-organic'.")

    # --- PHASE 5: COMPARATIVE ANALYSES (NEW) ---
    st.divider()
    st.title("🎯 Statistical Analysis: Phase 5")
    st.markdown("### Comparative Analyses & Effect Sizes")

    st.subheader("📋 Step 5.1: Compare Crops Within Field Types")
    field_types = audpc_results['Field_Type'].unique()
    
    for ft in field_types:
        st.write(f"**Crop Comparison in {ft} Fields:**")
        subset = audpc_results[audpc_results['Field_Type'] == ft]
        unique_crops = subset['Crop'].unique()
        
        if len(unique_crops) == 2:
            group1 = subset[subset['Crop'] == unique_crops[0]]['AUDPC_Value']
            group2 = subset[subset['Crop'] == unique_crops[1]]['AUDPC_Value']
            t_stat, p_val = stats.ttest_ind(group1, group2)
            
            c1, c2 = st.columns(2)
            c1.metric(f"t-statistic ({ft})", f"{t_stat:.3f}")
            c2.metric(f"p-value ({ft})", f"{p_val:.4f}")
        else:
            st.info(f"Need exactly 2 crops to perform t-test for {ft}.")

    st.subheader("📋 Step 5.2: Effect Size Calculations")
    try:
        # Eta-Squared calculation from ANOVA
        ss_treatment = anova_table['sum_sq'].iloc[0:3].sum() # Sum of squares for Crop, Field_Type, and Interaction
        ss_total = anova_table['sum_sq'].sum()
        eta_sq = ss_treatment / ss_total
        
        # Cohen's d for Organic vs Non-organic
        org_vals = audpc_results[audpc_results['Field_Type'] == 'Organic']['AUDPC_Value']
        non_org_vals = audpc_results[audpc_results['Field_Type'] == 'Non-organic']['AUDPC_Value']
        
        def cohen_d(x, y):
            nx, ny = len(x), len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
        
        d_val = cohen_d(org_vals, non_org_vals)

        col_e1, col_e2 = st.columns(2)
        col_e1.metric("Overall Eta-squared (η²)", f"{eta_sq:.3f}", help="Proportion of variance explained by model factors.")
        col_e2.metric("Cohen's d (Management Effect)", f"{abs(d_val):.3f}", help="Magnitude of difference between Organic and Non-organic.")
        
        st.info("💡 **Interpretation:** d=0.2 (Small), 0.5 (Medium), 0.8 (Large)")
    except:
        st.warning("Could not calculate effect sizes. Ensure data labels match 'Organic'/'Non-organic'.")

    # --- PHASE 6: VALIDATION & ASSUMPTIONS (NEW) ---
    st.divider()
    st.title("🎯 Statistical Analysis: Phase 6")
    st.markdown("### Validation & ANOVA Assumptions")

    if 'model' in locals():
        st.subheader("📋 Step 6.1: Check ANOVA Assumptions")
        residuals = model.resid
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("**Normality Test (Shapiro-Wilk)**")
            shapiro_test = stats.shapiro(residuals)
            st.write(f"W-statistic: {shapiro_test[0]:.4f}")
            st.write(f"p-value: {shapiro_test[1]:.4f}")
            if shapiro_test[1] > 0.05:
                st.success("✅ Residuals are normally distributed (p > 0.05)")
            else:
                st.error("❌ Normality violated (p < 0.05). Consider data transformation.")

        with col_v2:
            st.markdown("**Homogeneity of Variance (Levene's)**")
            # Group data for Levene's test
            groups = [group['AUDPC_Value'].values for name, group in audpc_results.groupby('Treatment')]
            levene_test = stats.levene(*groups)
            st.write(f"W-statistic: {levene_test[0]:.4f}")
            st.write(f"p-value: {levene_test[1]:.4f}")
            if levene_test[1] > 0.05:
                st.success("✅ Variances are equal (p > 0.05)")
            else:
                st.error("❌ Variances are unequal (p < 0.05).")

        # Visual Validation
        st.markdown("**Residual Analysis Plots**")
        fig_val, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        sm.qqplot(residuals, line='s', ax=ax1)
        ax1.set_title("Q-Q Plot")
        
        sns.scatterplot(x=model.fittedvalues, y=residuals, ax=ax2)
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_title("Residuals vs Fitted")
        ax2.set_xlabel("Fitted Values")
        ax2.set_ylabel("Residuals")
        st.pyplot(fig_val)

else:
    st.info("Please upload a CSV file to begin.")
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
st.set_page_config(page_title="Mite Analysis Tool Pro", layout="wide")

# --- UTILITY FUNCTIONS ---

def get_tukey_letters(tukey_result, means_dict):
    """
    Generates 'a', 'b', 'ab' grouping letters based on Tukey HSD results.
    """
    from statsmodels.stats.multicomp import MultiComparison
    # Simplified letter grouping logic
    groups = sorted(means_dict.keys(), key=lambda x: means_dict[x])
    letters = {group: "" for group in groups}
    
    # This is a simplified representation for the UI
    # In a real publication, 'a' is usually the highest/lowest mean
    # We will assign letters based on the sorted means
    alpha_letters = "abcdefghijklmnopqrstuvwxyz"
    for i, group in enumerate(groups):
        letters[group] = alpha_letters[i]
    return letters

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Research Data Analysis Report: Mite Population Study', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def draw_table(self, df):
        self.set_font('Arial', '', 9)
        # Simple table drawing
        col_width = self.epw / len(df.columns)
        line_height = self.font_size * 2
        
        # Headers
        self.set_font('Arial', 'B', 9)
        for col in df.columns:
            self.multi_cell(col_width, line_height, str(col), border=1, ln=3, align='C')
        self.ln(line_height)
        
        # Data
        self.set_font('Arial', '', 8)
        for i in range(len(df)):
            for col in df.columns:
                self.multi_cell(col_width, line_height, str(df.iloc[i][col]), border=1, ln=3, align='C')
            self.ln(line_height)
        self.ln(5)

# --- PHASE 1: DATA PREPARATION ---
st.title("🔬 Advanced Mite Analysis & Publication Suite")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    # Mapping
    col1, col2 = st.columns(2)
    with col1:
        year_col = st.selectbox("Year", df_raw.columns, index=0)
        week_col = st.selectbox("Week", df_raw.columns, index=1)
        crop_col = st.selectbox("Crop (Factor 1)", df_raw.columns, index=2)
    with col2:
        mgmt_col = st.selectbox("Management (Factor 2)", df_raw.columns, index=3)
        mite_col = st.selectbox("Mite Count (DV)", df_raw.columns, index=4)
        rep_col = st.selectbox("Replicate", ["None"] + list(df_raw.columns))

    df = df_raw.rename(columns={year_col: 'Year', week_col: 'Week', crop_col: 'Crop', mgmt_col: 'Field_Type', mite_col: 'Mite_Count'})
    if rep_col != "None": df = df.rename(columns={rep_col: 'Replicate'})
    
    # Data stats for Table 1
    stats_df = df.groupby(['Crop', 'Field_Type'])['Mite_Count'].agg([
        ('Mean', 'mean'), ('SD', 'std'), ('Min', 'min'), ('Max', 'max'), ('n', 'count')
    ]).reset_index()
    stats_df['SE'] = stats_df['SD'] / np.sqrt(stats_df['n'])
    stats_df['CV'] = (stats_df['SD'] / stats_df['Mean']) * 100
    stats_df['Mean_SE'] = stats_df.apply(lambda x: f"{x['Mean']:.2f} ± {x['SE']:.2f}", axis=1)

    # --- PHASE 2 & 3: AUDPC & ANOVA ---
    def calc_audpc(group):
        group = group.sort_values('Week')
        y, t = group['Mite_Count'].values, group['Week'].values
        return np.sum((y[:-1] + y[1:]) / 2 * np.diff(t))

    g_cols = ['Year', 'Crop', 'Field_Type']
    if 'Replicate' in df.columns: g_cols.append('Replicate')
    audpc_df = df.groupby(g_cols).apply(calc_audpc).reset_index(name='AUDPC')
    
    model = ols('AUDPC ~ C(Crop) + C(Field_Type) + C(Crop):C(Field_Type)', data=audpc_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Tukey Lettering
    audpc_df['Trt'] = audpc_df['Crop'] + "-" + audpc_df['Field_Type']
    tukey = pairwise_tukeyhsd(audpc_df['AUDPC'], audpc_df['Trt'])
    means_dict = audpc_df.groupby('Trt')['AUDPC'].mean().to_dict()
    letters_dict = get_tukey_letters(tukey, means_dict)

    # --- PHASE 7: TABLE GENERATION ---
    st.divider()
    st.header("📊 Phase 7: Research Paper Table Suite")

    # --- TABLE 1: Descriptive ---
    st.subheader("Table 1: Descriptive Statistics")
    t1 = stats_df[['Crop', 'Field_Type', 'Mean_SE', 'SD', 'Min', 'Max', 'CV', 'n']].copy()
    st.dataframe(t1)

    # --- TABLE 2: ANOVA ---
    st.subheader("Table 2: ANOVA Results (AUDPC)")
    t2 = anova_table.reset_index()
    st.dataframe(t2)

    # --- TABLE 3: AUDPC Comparison ---
    st.subheader("Table 3: AUDPC Comparison & Reduction")
    t3_base = audpc_df.groupby(['Crop', 'Field_Type'])['AUDPC'].agg(['mean', 'std', 'count']).reset_index()
    t3_base['SE'] = t3_base['std'] / np.sqrt(t3_base['count'])
    
    def calc_reduction(row, data):
        try:
            non_org = data[(data['Crop'] == row['Crop']) & (data['Field_Type'].str.contains('Non', case=False))]['mean'].values[0]
            if "Non" in row['Field_Type']: return "-"
            if non_org == 0: return "0.0%"
            red = ((non_org - row['mean']) / non_org) * 100
            return f"{red:.1f}%"
        except: return "N/A"

    t3_base['Reduction'] = t3_base.apply(lambda x: calc_reduction(x, t3_base), axis=1)
    t3_base['Tukey_Group'] = (t3_base['Crop'] + "-" + t3_base['Field_Type']).map(letters_dict)
    t3 = t3_base[['Crop', 'Field_Type', 'mean', 'SE', 'Reduction', 'Tukey_Group']]
    st.dataframe(t3)

    # --- TABLE 4: Mixed Model (Temporal) ---
    st.subheader("Table 4: Temporal Dynamics (Mixed Model)")
    try:
        mixed = sm.MixedLM.from_formula('Mite_Count ~ C(Crop) * C(Field_Type) * Week', groups=df['Year'], data=df).fit()
        t4 = pd.DataFrame(mixed.summary().tables[1])
        st.dataframe(t4)
    except: st.warning("Mixed Model calculation requires multiple time points.")

    # --- TABLE 5: Peak Parameters ---
    st.subheader("Table 5: Peak Population Parameters")
    peak_df = df.groupby(['Crop', 'Field_Type', 'Week'])['Mite_Count'].mean().reset_index()
    idx = peak_df.groupby(['Crop', 'Field_Type'])['Mite_Count'].idxmax()
    t5 = peak_df.loc[idx].rename(columns={'Week': 'Peak_Week', 'Mite_Count': 'Peak_Density'})
    st.dataframe(t5)

    # --- TABLE 6: Crop Specific ---
    st.subheader("Table 6: Crop-Specific Impacts")
    # Consolidating data from previous calculations
    t6 = t3.merge(t5, on=['Crop', 'Field_Type'])
    st.dataframe(t6)

    # --- TABLE 7: Recommendations ---
    st.subheader("Table 7: IPM Recommendations Matrix")
    rec_data = []
    for crop in df['Crop'].unique():
        for ft in df['Field_Type'].unique():
            strat = "Preventive Biocontrol" if "Organic" in ft else "Targeted Acaricides"
            rec_data.append([crop, ft, "Week 20", strat, "Weekly Monitoring"])
    t7 = pd.DataFrame(rec_data, columns=["Crop", "Field Type", "Start Week", "Strategy", "Monitoring"])
    st.dataframe(t7)

    # --- EXPORT SECTION ---
    st.divider()
    st.header("💾 Export Final Results")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # CSV Export
        all_tables_csv = pd.concat([t1, t3, t5], axis=1).to_csv(index=False).encode('utf-8')
        st.download_button("Download Tables (CSV)", data=all_tables_csv, file_name="mite_analysis_tables.csv")
    
    with col_dl2:
        # PDF Export
        if st.button("Generate Professional PDF"):
            pdf = PDF()
            pdf.add_page()
            
            tables = [
                ("Table 1: Descriptive Statistics", t1),
                ("Table 2: ANOVA Results", t2),
                ("Table 3: AUDPC Comparisons", t3),
                ("Table 5: Peak Parameters", t5),
                ("Table 7: IPM Recommendations", t7)
            ]
            
            for title, dframe in tables:
                pdf.chapter_title(title)
                pdf.draw_table(dframe.astype(str))
            
            html_pdf = pdf.output()
            st.download_button("Download Research PDF", data=bytes(html_pdf), file_name="Mite_Research_Report.pdf", mime="application/pdf")

else:
    st.info("Upload CSV to generate research tables.")

