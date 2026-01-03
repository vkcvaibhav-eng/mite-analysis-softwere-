import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- PHASE 2: PRIMARY STATISTICAL ANALYSES ---
st.divider()
st.title("🎯 Statistical Analysis: Phase 2")
st.markdown("### AUDPC Calculation & Primary ANOVA")

# We assume 'df_final' is available from Phase 1 logic
if 'df_final' in locals() or 'df_final' in globals():
    
    # --- STEP 2.1: AUDPC CALCULATION ---
    st.subheader("📋 Step 2.1: AUDPC Calculation")
    st.info("AUDPC summarizes the entire season's mite pressure into a single value for each plot/replicate.")

    def calculate_audpc(group):
        # Sort by week to ensure correct trapezoidal calculation
        group = group.sort_values('Week')
        y = group['Mite_Count'].values
        t = group['Week'].values
        
        # AUDPC Formula: Σ [(Yi + Yi+1)/2] × (ti+1 - ti)
        audpc_val = np.sum((y[:-1] + y[1:]) / 2 * np.diff(t))
        return audpc_val

    # Grouping by Year, Crop, Field_Type, and Replicate to get one AUDPC per plot
    grouping_cols = ['Year', 'Crop', 'Field_Type']
    if 'Replicate' in df_final.columns:
        grouping_cols.append('Replicate')

    audpc_results = df_final.groupby(grouping_cols).apply(calculate_audpc).reset_index()
    audpc_results.columns = grouping_cols + ['AUDPC_Value']

    st.write("Calculated AUDPC Values (First 10 rows):")
    st.dataframe(audpc_results.head(10).style.format({"AUDPC_Value": "{:.2f}"}))

    # --- STEP 2.2: TWO-WAY ANOVA ON AUDPC ---
    st.subheader("📋 Step 2.2: Two-Way ANOVA (Main Test)")
    
    # Building the Model
    # AUDPC ~ Crop + Field_Type + Crop:Field_Type
    model_formula = 'AUDPC_Value ~ C(Crop) + C(Field_Type) + C(Crop):C(Field_Type)'
    
    try:
        model = ols(model_formula, data=audpc_results).fit()
        anova_table = sm.stats.anova_lm(model, typ=2) # Typ 2 is standard for balanced/unbalanced designs

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
        # Convert Tukey results to DataFrame for better display
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        st.dataframe(tukey_df)
        
        # Highlight significant differences
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
    st.warning("Please complete Phase 1 and ensure data is properly loaded before proceeding to Phase 2.")
