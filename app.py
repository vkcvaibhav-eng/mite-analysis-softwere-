import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mite Analysis Pro", layout="wide")
sns.set_theme(style="whitegrid")
THRESHOLD = 2.0

# --- UTILITY FUNCTIONS ---
def get_tukey_letters(tukey_results):
    res_df = pd.DataFrame(data=tukey_results.summary().data[1:], columns=tukey_results.summary().data[0])
    groups = np.unique(np.concatenate([res_df['group1'], res_df['group2']]))
    alpha = "abcdefghijklmnopqrstuvwxyz"
    return {g: alpha[i % 26] for i, g in enumerate(sorted(groups))}

class PDF_Report(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'MITE RESEARCH PUBLICATION REPORT', 0, 1, 'C')
        self.ln(10)
    def add_table(self, title, df):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 10, title, 0, 1)
        self.set_font('Helvetica', '', 8)
        col_width = self.epw / len(df.columns)
        for col in df.columns: self.cell(col_width, 8, str(col), border=1)
        self.ln()
        for _, row in df.iterrows():
            for val in row: self.cell(col_width, 7, str(val), border=1)
            self.ln()
    def add_fig(self, title, fig):
        self.add_page()
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'C')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, dpi=300, bbox_inches='tight')
            self.image(tmp.name, x=10, w=190)
        os.unlink(tmp.name)

# --- MAIN APP ---
st.title("🔬 Mite Analysis & Publication Suite")
uploaded_file = st.file_uploader("Upload Mite Data (CSV)", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    
    # Selection UI
    col1, col2 = st.columns(2)
    with col1:
        yr = st.selectbox("Year", df_raw.columns, index=0)
        wk = st.selectbox("Week", df_raw.columns, index=1)
        cp = st.selectbox("Crop", df_raw.columns, index=2)
    with col2:
        mt = st.selectbox("Management", df_raw.columns, index=3)
        cnt = st.selectbox("Mite Count", df_raw.columns, index=4)
        rp = st.selectbox("Replicate", ["None"] + list(df_raw.columns))

    # Data Cleaning & Setup
    df = df_raw.rename(columns={yr:'Year', wk:'Week', cp:'Crop', mt:'Field_Type', cnt:'Mite_Count'})
    df['Treatment'] = df['Crop'] + " - " + df['Field_Type']
    if rp == "None": df['Replicate'] = 1
    else: df = df.rename(columns={rp: 'Replicate'})

    # DYNAMIC PALETTE (The Critical Fix)
    unique_treats = df['Treatment'].unique()
    palette = dict(zip(unique_treats, sns.color_palette("husl", len(unique_treats))))

    # Statistics Calculation
    audpc_df = df.groupby(['Year', 'Crop', 'Field_Type', 'Treatment', 'Replicate']).apply(
        lambda g: np.trapz(g.sort_values('Week')['Mite_Count'], g.sort_values('Week')['Week'])
    ).reset_index(name='AUDPC')

    model = ols('AUDPC ~ C(Crop) * C(Field_Type)', data=audpc_df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    tukey = pairwise_tukeyhsd(audpc_df['AUDPC'], audpc_df['Treatment'])
    tukey_map = get_tukey_letters(tukey)

    # --- THE 7 TABLES ---
    st.header("📋 Research Tables")
    t1 = df.groupby(['Treatment'])['Mite_Count'].agg(['mean', 'std']).reset_index() # Table 1
    t2 = anova.reset_index() # Table 2
    t3 = audpc_df.groupby('Treatment')['AUDPC'].mean().reset_index()
    t3['Group'] = t3['Treatment'].map(tukey_map) # Table 3
    
    try: # Table 4 Mixed Model
        mm = mixedlm("Mite_Count ~ Week * C(Field_Type)", df, groups=df["Year"]).fit()
        t4 = mm.summary().tables[1].reset_index()
    except: t4 = pd.DataFrame({"Status": ["Model variance too low"]})

    t5 = df.groupby('Treatment')['Mite_Count'].max().reset_index(name='Peak') # Table 5
    t6 = t3.copy() # Table 6 Impact Analysis
    t7 = pd.DataFrame([{"Treatment": t, "Strategy": "Organic IPM" if "Org" in t else "Standard IPM"} for t in unique_treats]) # Table 7

    tabs = st.tabs([f"Table {i}" for i in range(1, 8)])
    for i, table in enumerate([t1, t2, t3, t4, t5, t6, t7]):
        tabs[i].table(table)

    # --- THE 7 FIGURES ---
    st.header("📈 Research Figures")
    figs = {}

    # F1: Temporal Dynamics
    f1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x='Week', y='Mite_Count', hue='Treatment', palette=palette, marker='o', ax=ax1)
    ax1.axhline(THRESHOLD, color='red', linestyle='--')
    st.pyplot(f1); figs["Fig 1: Population Dynamics"] = f1

    # F2: AUDPC Comparison (Fixed)
    f2, ax2 = plt.subplots()
    sns.barplot(data=audpc_df, x='Crop', y='AUDPC', hue='Treatment', palette=palette, ax=ax2)
    st.pyplot(f2); figs["Fig 2: AUDPC Comparison"] = f2

    # F3: Peak Density
    f3, ax3 = plt.subplots()
    sns.barplot(data=t5, x='Treatment', y='Peak', palette=palette, ax=ax3)
    plt.xticks(rotation=45); st.pyplot(f3); figs["Fig 3: Peak Density"] = f3

    # F4: Threshold Duration
    f4, ax4 = plt.subplots()
    df['Above'] = df['Mite_Count'] > THRESHOLD
    sns.countplot(data=df[df['Above']], x='Treatment', palette=palette, ax=ax4)
    st.pyplot(f4); figs["Fig 4: Threshold duration"] = f4

    # F5: Panel Plot
    f5 = sns.relplot(data=df, x="Week", y="Mite_Count", hue="Field_Type", col="Crop", kind="line", palette="Set1").fig
    st.pyplot(f5); figs["Fig 5: Crop Comparison Panels"] = f5

    # F6: Intensity Heatmap
    f6, ax6 = plt.subplots()
    pivot = df.pivot_table(index='Treatment', columns='Week', values='Mite_Count')
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax6)
    st.pyplot(f6); figs["Fig 6: Population Heatmap"] = f6

    # F7: Seasonal Boxplots
    f7, ax7 = plt.subplots()
    sns.boxplot(data=df, x='Treatment', y='Mite_Count', palette=palette, ax=ax7)
    plt.xticks(rotation=45); st.pyplot(f7); figs["Fig 7: Population Distribution"] = f7

    # --- PDF EXPORT ---
    if st.button("Generate Complete Publication PDF Pack"):
        pdf = PDF_Report()
        pdf.add_page()
        pdf.add_table("Table 1: Descriptive Stats", t1)
        pdf.add_table("Table 3: AUDPC Stats", t3)
        for title, fig in figs.items(): pdf.add_fig(title, fig)
        
        output = pdf.output(dest='S').encode('latin-1')
        st.download_button("Download Research Pack", output, "Mite_Publication_Report.pdf", "application/pdf")
