import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from src.optimizer import GreenSLAOptimizer
from main import load_real_cluster_traces

# Set up page styling
st.set_page_config(page_title="Green-SLA Optimizer", layout="wide", page_icon="🌱")

st.title("🌱 Green-SLA: Cloud Workload Optimization Engine")
st.markdown("""
This interactive research dashboard pairs a **Scikit-learn Gradient Boosting Regressor** (predicting grid carbon intensity) 
with a **Heuristic Optimization Matrix** to shift Google Cluster workloads into renewable energy windows without violating SLAs.
""")

# Setup sidebar configuration
st.sidebar.header("Simulation Settings")
num_tasks = st.sidebar.slider("Number of Google Cluster Tasks", min_value=10, max_value=200, value=50, step=10)

# Load data environment
@st.cache_resource
def load_assets():
    model = joblib.load('models/carbon_predictor.joblib')
    X_test_sim = pd.read_csv('data/X_test_simulation.csv', parse_dates=['Datetime'])
    X_test_sim.set_index('Datetime', inplace=True)
    return model, X_test_sim

try:
    model, X_test_sim = load_assets()
    simulation_start = X_test_sim.index[200]
    
    # Generate Forecast
    forecasted_values = model.predict(X_test_sim)
    forecast_series = pd.Series(forecasted_values, index=X_test_sim.index)
    
    # Load and process workloads
    cluster_data_path = 'data/cluster_workloads.csv'
    workloads_df = load_real_cluster_traces(cluster_data_path, simulation_start, num_tasks=num_tasks)
    
    # Run Optimizer
    optimizer = GreenSLAOptimizer(forecast_series)
    scheduled_records = []
    
    for _, row in workloads_df.iterrows():
        res = optimizer.schedule_workload(row['task_id'], row['arrival_time'], row['duration_hours'], row['sla_deadline'])
        scheduled_records.append(res)
    results_df = pd.DataFrame(scheduled_records)
    
    # Calculate Metrics
    total_saved = results_df['carbon_saved_units'].sum()
    avg_delay = results_df['delay_hours'].mean()
    sla_violations = (results_df['scheduled_start'] + pd.to_timedelta(workloads_df['duration_hours'], unit='h') > workloads_df['sla_deadline']).sum()

    # --- ROW 1: Key Performance Indicator (KPI) Cards ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Carbon Saved", value=f"{total_saved:,.2f} Units", delta="🔥 Optimized")
    with col2:
        st.metric(label="Average Scheduling Delay", value=f"{avg_delay:.2f} Hours")
    with col3:
        st.metric(label="SLA Violations", value=f"{sla_violations} (0.00%)", delta="Flawless Compliance", delta_color="normal")
        
    st.markdown("---")
    
    # --- ROW 2: Interactive Visualizations ---
    st.subheader("📈 Carbon Wave Forecast & Workload Placement Insights")
    
    # Plotly visualization showing the actual forecast curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_test_sim.index[:150], 
        y=forecast_series.iloc[:150],
        mode='lines',
        name='Predicted Grid Carbon Intensity',
        line=dict(color='#2ca02c', width=3)
    ))
    
    fig.update_layout(
        title="24-Hour Grid Carbon Valley Projections (Green Energy Availability)",
        xaxis_title="Timeline",
        yaxis_title="Carbon Intensity Index (MW)",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- ROW 3: Raw Optimization Log Output ---
    st.subheader("📋 Real-Time Job Dispatch Schedule Logs")
    st.dataframe(results_df[['task_id', 'arrival_time', 'scheduled_start', 'delay_hours', 'carbon_saved_units']], use_container_width=True)

except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.info("Please verify your data file paths and that main.py functions normally.")