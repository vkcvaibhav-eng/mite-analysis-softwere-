import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# Set page config
st.set_page_config(page_title="Mite Population Statistical Analysis", layout="wide")

st.title("📊 Mite Population Statistical Analysis Roadmap")
st.markdown("This application performs a complete 6-phase statistical analysis for Okra and Brinjal mite data.")

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your Mite Data CSV", type=["csv"])

@st.cache_data
def load_data(file_obj):
    try:
        if file_obj is not None:
            df = pd.read_csv(file_obj)
        elif os.path.exists('CSV.csv'):
            df = pd.read_csv('CSV.csv')
        else:
            return None
        
        # Standardizing column names
        if 'Management' in df.columns:
            df = df.rename(columns={'Management': 'Field_Type'})
            
        # Clean string data to prevent grouping errors
        for col in ['Crop', 'Field_Type']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Required columns validation
        required_cols = ['Year', 'Week', 'Crop', 'Field_Type', 'Mite_Count']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {', '.join(missing)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data(uploaded_file)

if df is not None:
    # Organize into Tabs for better UX
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data & Stats", "ANOVA & AUDPC", "Temporal Patterns", "Crop Specific", "Assumptions"
    ])

    # --- PHASE 1: DATA PREPARATION ---
    with tab1:
        st.header("Phase 1: Data Preparation & Exploration")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Step 1.1: Dataset Overview")
            st.write(df.head())
            st.metric("Total Observations", len(df))

        with col2:
            st.subheader("Step 1.2: Descriptive Statistics")
            desc_stats = df.groupby(['Crop', 'Field_Type'])['Mite_Count'].agg([
                'mean', 'std', 'count', 'min', 'max'
            ]).reset_index()
            desc_stats['se'] = desc_stats['std'] / np.sqrt(desc_stats['count'])
            st.dataframe(desc_stats.style.format(precision=3))

    # --- PHASE 2: PRIMARY STATISTICAL ANALYSES ---
    with tab2:
        st.header("Phase 2: Primary Statistical Analyses")

        # Step 2.1: AUDPC Calculation
        def calculate_audpc(group):
            group = group.sort_values('Week')
            y = group['Mite_Count'].values
            t = group['Week'].values
            return np.trapz(y, t)

        audpc_df = df.groupby(['Year', 'Crop', 'Field_Type'])['Mite_Count'].apply(
            lambda x: calculate_audpc(df.loc[x.index])
        ).reset_index(name='AUDPC')

        st.subheader("Step 2.1: AUDPC Calculation")
        st.caption("Area Under Disease Progress Curve (Seasonal Pressure Summary)")
        st.dataframe(audpc_df.head())

        # Step 2.2: ANOVA
        st.subheader("Step 2.2: Two-Way ANOVA (Main Test)")
        # Using C() to ensure categorical treatment
        model = ols('AUDPC ~ C(Crop) * C(Field_Type) + C(Year)', data=audpc_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.write(anova_table)

        # Step 2.3: Post-Hoc
        st.subheader("Step 2.3: Tukey's HSD Post-Hoc")
        audpc_df['Combination'] = audpc_df['Crop'] + "-" + audpc_df['Field_Type']
        m_comp = pairwise_tukeyhsd(endog=audpc_df['AUDPC'], groups=audpc_df['Combination'], alpha=0.05)
        st.text(m_comp.summary())

    # --- PHASE 3: TEMPORAL PATTERN ANALYSES ---
    with tab3:
        st.header("Phase 3: Temporal Pattern Analyses")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.subheader("Step 3.1: Mixed Model (Repeated Measures)")
            try:
                mixed_model = mixedlm("Mite_Count ~ Crop * Field_Type * Week", df, groups=df["Year"]).fit()
                st.write(mixed_model.summary().tables[1])
            except:
                st.warning("Mixed Model failed to converge. Check if 'Year' has enough levels.")

        with col_t2:
            st.subheader("Step 3.2: Peak Week Identification")
            peak_analysis = df.loc[df.groupby(['Year', 'Crop', 'Field_Type'])['Mite_Count'].idxmax()]
            peak_summary = peak_analysis.groupby(['Crop', 'Field_Type'])[['Week', 'Mite_Count']].agg(['mean', 'sem'])
            st.dataframe(peak_summary)

        st.divider()
        st.subheader("Step 3.3: Economic Threshold Analysis")
        threshold_df = df[df['Mite_Count'] >= 2].groupby(['Year', 'Crop', 'Field_Type'])['Week'].min().reset_index()
        if not threshold_df.empty:
            threshold_summary = threshold_df.groupby(['Crop', 'Field_Type'])['Week'].agg(['mean', 'sem'])
            st.write("Mean week when population reaches threshold (2 mites/leaf):")
            st.dataframe(threshold_summary)
        else:
            st.info("No populations reached the economic threshold (>= 2).")

    # --- PHASE 4: CROP-SPECIFIC & COMPARATIVE ---
    with tab4:
        st.header("Phase 4 & 5: Crop Analysis & Effect Sizes")
        c1, c2 = st.columns(2)

        unique_crops = df['Crop'].unique()
        for i, crop_name in enumerate(unique_crops[:2]): 
            with [c1, c2][i]:
                st.subheader(f"{crop_name} Comparison")
                crop_data = audpc_df[audpc_df['Crop'] == crop_name]
                
                types = crop_data['Field_Type'].unique()
                # Flexibly check for Organic/Non-Organic regardless of case
                org_key = next((t for t in types if 'org' in t.lower() and 'non' not in t.lower()), None)
                non_org_key = next((t for t in types if 'non' in t.lower()), None)

                if org_key and non_org_key:
                    t_stat, p_val = stats.ttest_ind(
                        crop_data[crop_data['Field_Type'] == org_key]['AUDPC'],
                        crop_data[crop_data['Field_Type'] == non_org_key]['AUDPC']
                    )
                    st.write(f"**p-value:** {p_val:.4f}")
                    
                    means = crop_data.groupby('Field_Type')['AUDPC'].mean()
                    reduction = ((means[non_org_key] - means[org_key]) / means[non_org_key]) * 100
                    st.metric(f"{crop_name} Management Reduction", f"{reduction:.2f}%")
                else:
                    st.warning(f"Insufficient data for {crop_name} comparison.")

        st.subheader("Step 5.2: Effect Size (Eta-squared η²)")
        ss_total = anova_table['sum_sq'].sum()
        eta_sq = anova_table['sum_sq'] / ss_total
        st.write(eta_sq.dropna().to_frame(name="Effect Size (η²)"))

    # --- PHASE 6: VALIDATION & VISUALS ---
    with tab5:
        st.header("Phase 6: Validation & Visualizations")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Normality (Shapiro-Wilk)")
            shapiro_p = stats.shapiro(model.resid)[1]
            st.write(f"Residuals p-value: {shapiro_p:.4f}")
            if shapiro_p > 0.05:
                st.success("Data is normal")
            else:
                st.warning("Data is non-normal (Consider transformation)")

        with col_b:
            st.subheader("Homogeneity (Levene's)")
            groups = [group['AUDPC'].values for name, group in audpc_df.groupby('Combination')]
            levene_p = stats.levene(*groups)[1]
            st.write(f"Variances p-value: {levene_p:.4f}")
            if levene_p > 0.05:
                st.success("Variances are equal")
            else:
                st.warning("Variances are unequal")

        st.divider()
        st.subheader("Visual Analytics")
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Temporal Plot
        sns.lineplot(data=df, x='Week', y='Mite_Count', hue='Crop', style='Field_Type', ax=ax[0], marker='o')
        ax[0].set_title("Mite Population Over Time")

        # AUDPC Bar Plot
        sns.barplot(data=audpc_df, x='Crop', y='AUDPC', hue='Field_Type', ax=ax[1])
        ax[1].set_title("Total Seasonal Pressure (AUDPC)")

        st.pyplot(fig)

else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to begin. Your CSV should include: Year, Week, Crop, Field_Type, and Mite_Count.")
