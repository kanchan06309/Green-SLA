import os
import joblib
import numpy as np
import pandas as pd
from src.optimizer import GreenSLAOptimizer

def load_real_cluster_traces(file_path, start_date, num_tasks=50):
    """
    Loads and formats real Google Cluster Trace columns into our Green-SLA Optimizer.
    Uses 'collection_id' and 'instance_index' for unique IDs, 'average_usage' 
    as a duration proxy, and 'priority' to set realistic SLA deadline windows.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please verify the path.")
        
    # Read the dataset
    df_trace = pd.read_csv(file_path)
    
    # DATA CLEANING FIX: Force 'average_usage' to numeric values. 
    # errors='coerce' turns non-numeric strings safely into NaN values.
    df_trace['average_usage'] = pd.to_numeric(df_trace['average_usage'], errors='coerce')
    
    # Filter out rows where critical identification columns are missing
    df_trace = df_trace.dropna(subset=['collection_id', 'instance_index'])
    
    # RANDOM SAMPLING: Select a randomized batch of tasks on each execution
    df_trace = df_trace.sample(n=num_tasks, random_state=np.random.randint(1, 1000))
    
    # Map a continuous timeline from your grid forecast onto these tasks
    base_timestamps = pd.date_range(start=start_date, periods=len(df_trace), freq='1h')
    
    tasks = []
    for i in range(len(df_trace)):
        row = df_trace.iloc[i]
        
        # 1. Create a unique task ID using real Google trace columns
        task_id = f"GOOG_COLL_{int(row['collection_id'])}_INST_{int(row['instance_index'])}"
        
        # 2. Determine duration based on average usage metrics
        # If the value is NaN after cleaning, fall back to a default value of 0.5
        usage = row['average_usage'] if pd.notna(row['average_usage']) else 0.5
        if usage > 0.8:
            duration = 6
        elif usage > 0.5:
            duration = 4
        elif usage > 0.2:
            duration = 2
        else:
            duration = 1
            
        ts = base_timestamps[i]
        
        # 3. Read scheduling priority to determine strictness of SLA
        priority = row['priority'] if pd.notna(row['priority']) else 1
        if priority > 5:
            sla_window = 8   # High priority task: tight execution deadline
        else:
            sla_window = 18  # Low priority task: greater scheduling flexibility
            
        tasks.append({
            'task_id': task_id,
            'arrival_time': ts,
            'duration_hours': duration,
            'sla_deadline': ts + pd.Timedelta(hours=sla_window)
        })
        
    return pd.DataFrame(tasks)

if __name__ == "__main__":
    print("--- Initializing Green-SLA Simulation Pipeline ---")
    
    # Paths verification
    model_path = 'models/carbon_predictor.joblib'
    sim_data_path = 'data/X_test_simulation.csv'
    cluster_data_path = 'data/cluster_workloads.csv'
    
    # 1. Load ML Model & Test data environment
    model = joblib.load(model_path)
    X_test_sim = pd.read_csv(sim_data_path, parse_dates=['Datetime'])
    X_test_sim.set_index('Datetime', inplace=True)
    
    # Generate continuous forecast using our ML model predictions
    forecasted_values = model.predict(X_test_sim)
    forecast_series = pd.Series(forecasted_values, index=X_test_sim.index)
    
    # 2. Instantiate Optimizer
    optimizer = GreenSLAOptimizer(forecast_series)
    
    # 3. Create simulated workloads from real Google Cluster Traces
    # Start the simulation deep into the test set to ensure forecast coverage
    simulation_start = X_test_sim.index[200]  
    
    print(f"Parsing traces from {cluster_data_path}...")
    workloads_df = load_real_cluster_traces(cluster_data_path, simulation_start, num_tasks=50)
    
    # 4. Run Optimization Engine
    scheduled_records = []
    print("Running multi-objective heuristic optimization matrix...")
    for _, row in workloads_df.iterrows():
        result = optimizer.schedule_workload(
            row['task_id'], row['arrival_time'], row['duration_hours'], row['sla_deadline']
        )
        scheduled_records.append(result)
        
    results_df = pd.DataFrame(scheduled_records)
    
    # 5. Compute Final Metrics
    total_saved = results_df['carbon_saved_units'].sum()
    avg_delay = results_df['delay_hours'].mean()
    
    # Verify if any task ended up executing past its assigned SLA deadline
    sla_violations = (results_df['scheduled_start'] + pd.to_timedelta(workloads_df['duration_hours'], unit='h') > workloads_df['sla_deadline']).sum()
    
    # Compute percentage safety metric
    violation_rate = (sla_violations / len(workloads_df)) * 100
    
    print("\n================ EXPERIMENTAL RESULTS ================")
    print(f"Total Carbon Emissions Mitigated : {total_saved:,.2f} Units")
    print(f"Average Workload Scheduling Delay: {avg_delay:.2f} Hours")
    print(f"Total Detected SLA Violations    : {sla_violations} ({violation_rate:.2f}%)")
    print("======================================================")