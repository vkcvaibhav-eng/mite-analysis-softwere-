import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Mite Statistical Analysis Roadmap", layout="wide")

st.title("🔬 Mite Population Professional Analysis")
st.markdown("Implementation of the **FINAL STATISTICAL ANALYSIS ROADMAP** (Phases 1-6).")

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload your CSV.csv file", type=["csv"])

if uploaded_file is not None:
    # PHASE 1: DATA PREPARATION
    df = pd.read_csv(uploaded_file)
    
    # Roadmap Step 1.1: Rename Management to Field_Type to match roadmap
    if 'Management' in df.columns:
        df = df.rename(columns={'Management': 'Field_Type'})
    
    required_cols = ['Year', 'Week', 'Crop', 'Field_Type', 'Mite_Count']
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"Error: CSV missing columns. Required: {required_cols}")
    else:
        # Create Tabs for the 6 Phases
        tab_prep, tab_primary, tab_temporal, tab_crop_spec, tab_comparative, tab_validation = st.tabs([
            "Phase 1: Prep", "Phase 2: AUDPC & ANOVA", "Phase 3: Temporal", 
            "Phase 4: Crop Specific", "Phase 5: Comparative", "Phase 6: Validation"
        ])

        # --- PHASE 1: PREPARATION & EXPLORATION ---
        with tab_prep:
            st.header("Step 1.2: Descriptive Statistics")
            stats_df = df.groupby(['Crop', 'Field_Type'])['Mite_Count'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()
            stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
            st.dataframe(stats_df.style.format(precision=3))
            
            # Overview Graph
            st.subheader("Visual Overview")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df, x='Crop', y='Mite_Count', hue='Field_Type', ax=ax1)
            st.pyplot(fig1)

        # --- PHASE 2: PRIMARY STATISTICAL ANALYSES ---
        with tab_primary:
            st.header("Step 2.1: AUDPC Calculation")
            # AUDPC Function: Σ [(Yi + Yi+1)/2] × (ti+1 - ti)
            def get_audpc(group):
                group = group.sort_values('Week')
                return np.trapz(group['Mite_Count'], group['Week'])

            audpc_df = df.groupby(['Year', 'Crop', 'Field_Type'])['Mite_Count'].apply(
                lambda x: get_audpc(df.loc[x.index])
            ).reset_index(name='AUDPC')
            st.write("Calculated AUDPC values per Year/Crop/Management:")
            st.dataframe(audpc_df)

            st.header("Step 2.2: Two-Way ANOVA on AUDPC")
            # AUDPC = Crop + Field_Type + Interaction + Year
            model = ols('AUDPC ~ C(Crop) * C(Field_Type) + C(Year)', data=audpc_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.write(anova_table)

            st.header("Step 2.3: Tukey's HSD Post-Hoc")
            audpc_df['Group'] = audpc_df['Crop'] + "_" + audpc_df['Field_Type']
            tukey = pairwise_tukeyhsd(endog=audpc_df['AUDPC'], groups=audpc_df['Group'], alpha=0.05)
            st.write(tukey.summary())

        # --- PHASE 3: TEMPORAL PATTERN ANALYSES ---
        with tab_temporal:
            st.header("Step 3.1: Mixed Model Repeated Measures")
            try:
                # Year as Random Factor
                md = mixedlm("Mite_Count ~ Crop * Field_Type * Week", df, groups=df["Year"])
                mdf = md.fit()
                st.write(mdf.summary().tables[1])
            except:
                st.warning("Mixed model failed. This usually happens if there isn't enough variance in 'Year'.")

            col1, col2 = st.columns(2)
            with col1:
                st.header("Step 3.2: Peak Week")
                peak_df = df.loc[df.groupby(['Year', 'Crop', 'Field_Type'])['Mite_Count'].idxmax()]
                peak_summary = peak_df.groupby(['Crop', 'Field_Type'])['Week'].agg(['mean', 'sem'])
                st.write(peak_summary)
            with col2:
                st.header("Step 3.3: Threshold (Count >= 2)")
                thresh = df[df['Mite_Count'] >= 2].groupby(['Year', 'Crop', 'Field_Type'])['Week'].min().reset_index()
                st.write(thresh.groupby(['Crop', 'Field_Type'])['Week'].agg(['mean', 'sem']))

        # --- PHASE 4: CROP-SPECIFIC ANALYSES ---
        with tab_crop_spec:
            for crop in df['Crop'].unique():
                st.subheader(f"Analysis for {crop}")
                c_data = audpc_df[audpc_df['Crop'] == crop]
                m_org = c_data[c_data['Field_Type'] == 'Organic']['AUDPC'].mean()
                m_non = c_data[c_data['Field_Type'] == 'Non organic']['AUDPC'].mean()
                reduction = ((m_non - m_org) / m_non) * 100
                st.metric(f"{crop} Reduction via Organic", f"{reduction:.2f}%")
                
                fig_c, ax_c = plt.subplots(figsize=(5, 3))
                sns.barplot(data=c_data, x='Field_Type', y='AUDPC', ax=ax_c)
                st.pyplot(fig_c)

        # --- PHASE 5: COMPARATIVE ---
        with tab_comparative:
            st.header("Step 5.2: Effect Size (Eta-Squared)")
            ss_total = anova_table['sum_sq'].sum()
            eta_sq = anova_table['sum_sq'] / ss_total
            st.write("Practical Significance (η²):")
            st.write(eta_sq.dropna())

        # --- PHASE 6: VALIDATION ---
        with tab_validation:
            st.header("Step 6.1: ANOVA Assumptions")
            shapiro_p = stats.shapiro(model.resid)[1]
            st.write(f"Normality (Shapiro-Wilk) p-value: {shapiro_p:.4f}")
            if shapiro_p > 0.05: st.success("Assumption Met: Data is Normal")
            else: st.error("Assumption Failed: Data is Non-Normal")
            
            # Visualizing Temporal Trends
            st.subheader("Seasonal Population Dynamics")
            fig_final, ax_final = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=df, x='Week', y='Mite_Count', hue='Crop', style='Field_Type', marker='o')
            st.pyplot(fig_final)

else:
    st.info("👋 Welcome! Please upload your 'CSV.csv' file in the sidebar to generate the roadmap analysis.")
