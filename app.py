import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

# Set page layout and theme
st.set_page_config(
    page_title="Mite Population Analysis System",
    page_icon="🕷️",
    layout="wide"
)

# Custom CSS to mimic the React app's styling
st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .treatment-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 5px solid #16a34a;
        margin-bottom: 20px;
    }
    .treatment-title { color: #15803d; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px; }
    </style>
""", unsafe_content_type=True)

st.title("🕷️ Mite Population Statistical Analysis System")

# --- DATA ANALYSIS LOGIC ---
def perform_analysis(df):
    # Ensure correct data types
    df['Week'] = pd.to_numeric(df['Week'])
    df['Mite_Count'] = pd.to_numeric(df['Mite_Count'])
    
    # Sort by week for correct AUDPC calculation
    df = df.sort_values(['Crop', 'Management', 'Week'])
    
    results = []
    groups = df.groupby(['Crop', 'Management'])
    
    for (crop, mgmt), group in groups:
        counts = group['Mite_Count'].values
        
        # AUDPC Calculation (Trapezoidal Rule for unit intervals)
        audpc = 0
        if len(counts) > 1:
            audpc = ((counts[:-1] + counts[1:]) / 2).sum()
        
        # Peak Statistics
        peak_val = group['Mite_Count'].max()
        peak_wk = group.loc[group['Mite_Count'] == peak_val, 'Week'].iloc[0]
        
        # Threshold Week (First week >= 2)
        thresh_df = group[group['Mite_Count'] >= 2]
        thresh_wk = thresh_df['Week'].min() if not thresh_df.empty else 0
        
        results.append({
            "treatment": f"{crop}-{mgmt}",
            "crop": crop,
            "field_type": mgmt,
            "audpc": round(float(audpc), 2),
            "peak_week": int(peak_wk),
            "peak_density": round(float(peak_val), 2),
            "threshold_week": int(thresh_wk)
        })
    
    # Weekly mean for plotting
    plot_df = df.pivot_table(index='Week', columns=['Crop', 'Management'], values='Mite_Count', aggfunc='mean')
    plot_df.columns = [f"{c}-{m}" for c, m in plot_df.columns]
    plot_df = plot_df.reset_index()
    
    return pd.DataFrame(results), plot_df

# --- SIDEBAR & UPLOAD ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    st.info("Required columns: Week, Crop, Management, Mite_Count")

if uploaded_file:
    # Read and clean data
    raw_df = pd.read_csv(uploaded_file)
    # Map 'Management' to 'field_type' logic if necessary
    audpc_results, plot_data = perform_analysis(raw_df)

    # --- TABS ---
    tab_labels = ["📤 Upload", "📊 Summary", "📈 Graphs", "📋 Tables"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

    with tab1:
        st.success("File uploaded successfully!")
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.write(f"Total Rows: {len(raw_df)}")

    with tab2:
        st.subheader("Analysis Summary")
        cols = st.columns(2)
        for i, row in audpc_results.iterrows():
            with cols[i % 2]:
                st.markdown(f"""
                <div class="treatment-card">
                    <div class="treatment-title">{row['treatment']}</div>
                    <p><b>AUDPC:</b> {row['audpc']}</p>
                    <p><b>Peak:</b> {row['peak_density']} (Week {row['peak_week']})</p>
                    <p><b>Threshold Week:</b> {row['threshold_week']}</p>
                </div>
                """, unsafe_content_type=True)

    with tab3:
        st.subheader("Population Trends")
        
        # Replicating the specific colors from your React app
        colors = {
            "Okra-Organic": "#16a34a",
            "Okra-Non organic": "#dc2626",
            "Brinjal-Organic": "#2563eb",
            "Brinjal-Non organic": "#f59e0b"
        }
        
        fig = go.Figure()
        for col in plot_data.columns:
            if col != 'Week':
                fig.add_trace(go.Scatter(
                    x=plot_data['Week'], 
                    y=plot_data[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(color=colors.get(col, None), width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Week",
            yaxis_title="Mean Mite Count",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Statistical Data")
        st.dataframe(audpc_results, use_container_width=True)
        
        csv = audpc_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results CSV",
            data=csv,
            file_name="Mite_Analysis_Results.csv",
            mime="text/csv"
        )
else:
    st.warning("Please upload a CSV file in the sidebar to view the analysis.")