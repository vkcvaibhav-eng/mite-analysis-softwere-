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

# --- SIDEBAR: CSV UPLOAD ---
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# --- PHASE 1: DATA PREPARATION & EXPLORATION ---
@st.cache_data
def load_data(file):
    try:
        if file is not None:
            df = pd.read_csv(file)
        else:
            # Try to load the local file if no upload exists
            df = pd.read_csv('CSV.csv')
        
        # Standardizing column names
        if 'Management' in df.columns:
            df = df.rename(columns={'Management': 'Field_Type'})
            
        return df
    except FileNotFoundError:
        st.error("Default 'CSV.csv' not found. Please upload a file via the sidebar.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = load_data(uploaded_file)

if df is not None:
    # Ensure necessary columns exist
    required_cols = {'Year', 'Week', 'Crop', 'Field_Type', 'Mite_Count'}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing columns! Please ensure your CSV has: {required_cols}")
    else:
        st.header("Phase 1: Data Preparation & Exploration")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Step 1.1: Dataset Overview")
            st.write(df.head())
            st.write(f"**Total Observations:** {len(df)}")

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

        # Step 3.3: Economic Threshold (Count >= 2)
        st.subheader("Step 3.2: Economic Threshold Analysis")
        threshold_df = df[df['Mite_Count'] >= 2].groupby(['Year', 'Crop', 'Field_Type'])['Week'].min().reset_index()
        threshold_summary = threshold_df.groupby(['Crop', 'Field_Type'])['Week'].agg(['mean', 'sem'])
        st.write("Mean week when population reaches threshold (2 mites/leaf):")
        st.dataframe(threshold_summary)

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
