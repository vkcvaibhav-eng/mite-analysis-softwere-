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
st.markdown("Upload your mite data to perform a complete 6-phase statistical analysis.")

# --- SIDEBAR: CSV UPLOAD ---
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your Mite Data (CSV)", type=["csv"])

# --- PHASE 1: DATA PREPARATION & EXPLORATION ---
@st.cache_data
def load_data(file_source):
    try:
        if file_source is not None:
            # Load the file uploaded by the user
            df = pd.read_csv(file_source)
        else:
            # Fallback to local file if it exists
            df = pd.read_csv('CSV.csv')
        
        # Standardizing column names (Management -> Field_Type)
        if 'Management' in df.columns:
            df = df.rename(columns={'Management': 'Field_Type'})
            
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return None

df = load_data(uploaded_file)

if df is not None:
    # Check if required columns exist
    required_cols = {'Year', 'Week', 'Crop', 'Field_Type', 'Mite_Count'}
    if not required_cols.issubset(df.columns):
        st.error(f"⚠️ **Missing Columns!** Your CSV must include: {', '.join(required_cols)}")
        st.info("Check if your 'Management' column is spelled correctly; the app will rename it to 'Field_Type' automatically.")
    else:
        st.header("Phase 1: Data Preparation & Exploration")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Step 1.1: Dataset Overview")
            st.write(df.head())
            st.write(f"**Total Rows:** {len(df)}")

        with col2:
            st.subheader("Step 1.2: Descriptive Statistics")
            desc_stats = df.groupby(['Crop', 'Field_Type'])['Mite_Count'].agg([
                'mean', 'std', 'count', 'min', 'max'
            ]).reset_index()
            desc_stats['se'] = desc_stats['std'] / np.sqrt(desc_stats['count'])
            st.dataframe(desc_stats.style.format(precision=3))

        # --- PHASE 2: PRIMARY STATISTICAL ANALYSES ---
        st.divider()
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
        st.dataframe(audpc_df.head())

        # Step 2.2: Two-Way ANOVA
        st.subheader("Step 2.2: Two-Way ANOVA (AUDPC)")
        model = ols('AUDPC ~ C(Crop) * C(Field_Type) + C(Year)', data=audpc_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.write(anova_table)

        # Step 2.3: Post-Hoc Tests
        st.subheader("Step 2.3: Tukey's HSD Post-Hoc")
        audpc_df['Combination'] = audpc_df['Crop'] + "-" + audpc_df['Field_Type']
        m_comp = pairwise_tukeyhsd(endog=audpc_df['AUDPC'], groups=audpc_df['Combination'], alpha=0.05)
        st.write(m_comp.summary())

        # --- PHASE 3: TEMPORAL PATTERN ANALYSES ---
        st.divider()
        st.header("Phase 3: Temporal Pattern Analyses")
        
        mixed_model = mixedlm("Mite_Count ~ Crop * Field_Type * Week", df, groups=df["Year"]).fit()
        st.write("**Mixed Model Results:**")
        st.write(mixed_model.summary().tables[1])

        # --- VISUALIZATIONS ---
        st.divider()
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
    st.info("👈 Please upload a CSV file in the sidebar to start the analysis.")
