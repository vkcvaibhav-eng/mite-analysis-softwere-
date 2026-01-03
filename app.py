import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set page config
st.set_page_config(page_title="Mite Population Statistical Analysis", layout="wide")

st.title("📊 Mite Population Statistical Analysis Roadmap")
st.markdown("This application performs a complete 6-phase statistical analysis for Okra and Brinjal mite data.")

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your Mite Data CSV", type=["csv"])

# --- PHASE 1: DATA PREPARATION & EXPLORATION ---
@st.cache_data
def load_data(file_obj):
    try:
        if file_obj is not None:
            df = pd.read_csv(file_obj)
        else:
            # Default to CSV.csv if no file is uploaded
            df = pd.read_csv('CSV.csv')
        
        # Standardizing column names
        if 'Management' in df.columns:
            df = df.rename(columns={'Management': 'Field_Type'})
            
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
    st.header("Phase 1: Data Preparation & Exploration")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Step 1.1: Dataset Overview")
        st.write(df.head())
        st.write(f"Total Observations: {len(df)}")

    with col2:
        st.subheader("Step 1.2: Descriptive Statistics")
        desc_stats = df.groupby(['Crop', 'Field_Type'])['Mite_Count'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).reset_index()
        desc_stats['se'] = desc_stats['std'] / np.sqrt(desc_stats['count'])
        st.dataframe(desc_stats.style.format(precision=3))

    # --- PHASE 2: PRIMARY STATISTICAL ANALYSES ---
    st.header("Phase 2: Primary Statistical Analyses")

    # Step 2.1: AUDPC Calculation
    def calculate_audpc(group):
        group = group.sort_values('Week')
        y = group['Mite_Count'].values
        t = group['Week'].values
        audpc = np.trapz(y, t) # Numerical integration
        return audpc

    # Group by factors and calculate AUDPC
    audpc_df = df.groupby(['Year', 'Crop', 'Field_Type'])['Mite_Count'].apply(
        lambda x: calculate_audpc(df.loc[x.index])
    ).reset_index(name='AUDPC')

    st.subheader("Step 2.1: AUDPC Calculation")
    st.write("Area Under Disease Progress Curve (Seasonal Pressure Summary)")
    st.dataframe(audpc_df.head())

    # Step 2.2: Two-Way ANOVA on AUDPC
    st.subheader("Step 2.2: Two-Way ANOVA (Main Test)")
    model = ols('AUDPC ~ C(Crop) * C(Field_Type) + C(Year)', data=audpc_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.write(anova_table)

    # Step 2.3: Post-Hoc Tests
    st.subheader("Step 2.3: Tukey's HSD Post-Hoc")
    audpc_df['Combination'] = audpc_df['Crop'] + "-" + audpc_df['Field_Type']
    m_comp = pairwise_tukeyhsd(endog=audpc_df['AUDPC'], groups=audpc_df['Combination'], alpha=0.05)
    st.write(m_comp.summary())

    # --- PHASE 3: TEMPORAL PATTERN ANALYSES ---
    st.header("Phase 3: Temporal Pattern Analyses")

    # Step 3.1: Mixed Model
    st.subheader("Step 3.1: Mixed Model Repeated Measures")
    mixed_model = mixedlm("Mite_Count ~ Crop * Field_Type * Week", df, groups=df["Year"]).fit()
    st.write(mixed_model.summary().tables[1])

    # Step 3.2: Peak Week Identification
    st.subheader("Step 3.2: Peak Week Identification")
    peak_analysis = df.loc[df.groupby(['Year', 'Crop', 'Field_Type'])['Mite_Count'].idxmax()]
    peak_summary = peak_analysis.groupby(['Crop', 'Field_Type'])[['Week', 'Mite_Count']].agg(['mean', 'sem'])
    st.dataframe(peak_summary)

    # Step 3.3: Economic Threshold (Count >= 2)
    st.subheader("Step 3.3: Economic Threshold Analysis")
    threshold_df = df[df['Mite_Count'] >= 2].groupby(['Year', 'Crop', 'Field_Type'])['Week'].min().reset_index()
    threshold_summary = threshold_df.groupby(['Crop', 'Field_Type'])['Week'].agg(['mean', 'sem'])
    st.write("Mean week when population reaches threshold (2 mites/leaf):")
    st.dataframe(threshold_summary)

    # --- PHASE 4: CROP-SPECIFIC ANALYSES ---
    st.header("Phase 4: Crop-Specific Analyses")
    c1, c2 = st.columns(2)

    unique_crops = df['Crop'].unique()
    for i, crop_name in enumerate(unique_crops[:2]): # Handle up to 2 crops for columns
        with [c1, c2][i]:
            st.subheader(f"Step 4.{i+1}: {crop_name} Analysis")
            crop_data = audpc_df[audpc_df['Crop'] == crop_name]
            
            # Check if we have both types for t-test
            types = crop_data['Field_Type'].unique()
            if 'Organic' in types and 'Non organic' in types:
                t_stat, p_val = stats.ttest_ind(
                    crop_data[crop_data['Field_Type'] == 'Organic']['AUDPC'],
                    crop_data[crop_data['Field_Type'] == 'Non organic']['AUDPC']
                )
                st.write(f"Organic vs Non-organic p-value: {p_val:.4f}")
                
                # % Reduction
                means = crop_data.groupby('Field_Type')['AUDPC'].mean()
                reduction = ((means['Non organic'] - means['Organic']) / means['Non organic']) * 100
                st.metric(f"{crop_name} Organic Reduction", f"{reduction:.2f}%")
            else:
                st.warning(f"Not enough variety in Field_Type for {crop_name} comparison.")

    # --- PHASE 5: COMPARATIVE ANALYSES ---
    st.header("Phase 5: Comparative Analyses")
    st.subheader("Step 5.2: Effect Size Calculations")
    ss_total = anova_table['sum_sq'].sum()
    eta_sq = anova_table['sum_sq'] / ss_total
    st.write("Eta-squared (η²) for ANOVA factors:")
    st.write(eta_sq.dropna())

    # --- PHASE 6: VALIDATION & ASSUMPTIONS ---
    st.header("Phase 6: Validation & Assumptions")
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

    # --- VISUALIZATIONS ---
    st.header("Visual Analytics")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Temporal Plot
    sns.lineplot(data=df, x='Week', y='Mite_Count', hue='Crop', style='Field_Type', ax=ax[0], marker='o')
    ax[0].set_title("Mite Population Over Time")

    # AUDPC Bar Plot
    sns.barplot(data=audpc_df, x='Crop', y='AUDPC', hue='Field_Type', ax=ax[1])
    ax[1].set_title("Total Seasonal Pressure (AUDPC)")

    st.pyplot(fig)
else:
    st.info("Please upload a CSV file via the sidebar to begin analysis.")
