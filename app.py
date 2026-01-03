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

st.title("📊 Professional Mite Analysis Dashboard")
st.markdown("Upload your data to execute the **Final Statistical Analysis Roadmap**.")

# --- SIDEBAR: FILE UPLOADER ---
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Phase 1: Data Preparation
    df = pd.read_csv(uploaded_file)
    
    # Map 'Management' to 'Field_Type' if necessary as per roadmap
    if 'Management' in df.columns:
        df = df.rename(columns={'Management': 'Field_Type'})
    
    # Check for required columns
    required_cols = ['Year', 'Week', 'Crop', 'Field_Type', 'Mite_Count']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns! Your CSV must have: {required_cols}")
    else:
        # Layout Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Data & Descriptive", 
            "📈 AUDPC & ANOVA", 
            "🕒 Temporal & Peak", 
            "🌿 Crop Specific", 
            "🔍 Assumptions"
        ])

        with tab1:
            st.header("Phase 1: Data Preparation & Exploration")
            st.subheader("Step 1.1: Dataset Preview")
            st.dataframe(df.head(10))
            
            st.subheader("Step 1.2: Descriptive Statistics")
            desc_stats = df.groupby(['Crop', 'Field_Type'])['Mite_Count'].agg([
                'mean', 'std', 'count', 'min', 'max'
            ]).reset_index()
            desc_stats['se'] = desc_stats['std'] / np.sqrt(desc_stats['count'])
            st.write("Summary Statistics per Treatment:")
            st.dataframe(desc_stats.style.format(precision=3))

        with tab2:
            st.header("Phase 2: Primary Statistical Analyses")
            
            # Step 2.1: AUDPC Calculation
            def calculate_audpc(group):
                group = group.sort_values('Week')
                # trapz handles the formula: Σ [(Yi + Yi+1)/2] × (ti+1 - ti)
                return np.trapz(group['Mite_Count'], group['Week'])

            audpc_df = df.groupby(['Year', 'Crop', 'Field_Type'])['Mite_Count'].apply(
                lambda x: calculate_audpc(df.loc[x.index])
            ).reset_index(name='AUDPC')

            st.subheader("Step 2.1: AUDPC Results")
            st.dataframe(audpc_df)

            # Step 2.2: Two-Way ANOVA
            st.subheader("Step 2.2: Two-Way ANOVA on AUDPC")
            try:
                model_formula = 'AUDPC ~ C(Crop) * C(Field_Type) + C(Year)'
                model = ols(model_formula, data=audpc_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.write(anova_table)

                # Step 2.3: Post-Hoc
                st.subheader("Step 2.3: Tukey's HSD Post-Hoc")
                audpc_df['Combination'] = audpc_df['Crop'] + " (" + audpc_df['Field_Type'] + ")"
                m_comp = pairwise_tukeyhsd(endog=audpc_df['AUDPC'], groups=audpc_df['Combination'], alpha=0.05)
                st.write(m_comp.summary())
            except Exception as e:
                st.warning(f"ANOVA could not be calculated. Ensure you have enough replicates/years. Error: {e}")

        with tab3:
            st.header("Phase 3: Temporal Pattern Analyses")
            
            # Visualizing the trend
            fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=df, x='Week', y='Mite_Count', hue='Crop', style='Field_Type', marker='o', ax=ax_trend)
            st.pyplot(fig_trend)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Peak Week Analysis")
                peak_idx = df.groupby(['Year', 'Crop', 'Field_Type'])['Mite_Count'].idxmax()
                peak_df = df.loc[peak_idx]
                peak_sum = peak_df.groupby(['Crop', 'Field_Type'])['Week'].agg(['mean', 'std']).rename(columns={'mean': 'Avg Peak Week'})
                st.dataframe(peak_sum)

            with col_b:
                st.subheader("Economic Threshold (>= 2)")
                threshold_df = df[df['Mite_Count'] >= 2].groupby(['Year', 'Crop', 'Field_Type'])['Week'].min().reset_index()
                if not threshold_df.empty:
                    thresh_sum = threshold_df.groupby(['Crop', 'Field_Type'])['Week'].agg(['mean', 'std'])
                    st.dataframe(thresh_sum)
                else:
                    st.info("Mite count never reached threshold of 2.")

        with tab4:
            st.header("Phase 4 & 5: Crop-Specific & Comparative")
            crops = df['Crop'].unique()
            for crop in crops:
                st.subheader(f"Analysis for {crop}")
                c_data = audpc_df[audpc_df['Crop'] == crop]
                means = c_data.groupby('Field_Type')['AUDPC'].mean()
                
                if len(means) == 2:
                    reduction = ((means['Non organic'] - means['Organic']) / means['Non organic']) * 100
                    st.metric(f"{crop} Organic Reduction %", f"{reduction:.2f}%")
                
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                sns.barplot(data=c_data, x='Field_Type', y='AUDPC', ax=ax_bar)
                st.pyplot(fig_bar)

        with tab5:
            st.header("Phase 6: Validation of Assumptions")
            if 'model' in locals():
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Normality (Shapiro-Wilk)**")
                    shapiro_p = stats.shapiro(model.resid)[1]
                    st.write(f"p-value: {shapiro_p:.4f}")
                    st.info("Goal: p > 0.05 for normal distribution.")
                
                with col2:
                    st.write("**Homogeneity (Levene's)**")
                    groups = [group['AUDPC'].values for name, group in audpc_df.groupby('Combination')]
                    levene_p = stats.levene(*groups)[1]
                    st.write(f"p-value: {levene_p:.4f}")
                    st.info("Goal: p > 0.05 for equal variances.")

else:
    st.info("Please upload your CSV file in the sidebar to begin the analysis.")
    st.image("https://via.placeholder.com/800x400.png?text=Awaiting+Data+Upload", use_column_width=True)



[Image of a data analysis dashboard]
