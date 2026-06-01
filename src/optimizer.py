import numpy as np
import pandas as pd

class GreenSLAOptimizer:
    def __init__(self, forecast_series):
        """
        Initialize with a series of predicted carbon intensities.
        forecast_series should use a DatetimeIndex.
        """
        self.forecast = forecast_series

    def schedule_workload(self, task_id, arrival_time, duration_hours, sla_deadline_time):
        """
        Calculates the optimal execution window to minimize carbon footprint.
        """
        arrival_time = pd.to_datetime(arrival_time)
        sla_deadline_time = pd.to_datetime(sla_deadline_time)
        
        # Determine the latest possible start time to meet the SLA
        latest_start_time = sla_deadline_time - pd.Timedelta(hours=duration_hours)
        
        # Generate a list of all valid hourly starting points in our window
        available_slots = pd.date_range(start=arrival_time, end=latest_start_time, freq='h')
        
        best_start_time = arrival_time
        lowest_carbon_cost = float('inf')
        
        # Heuristic optimization scan
        for slot in available_slots:
            execution_window = pd.date_range(start=slot, periods=duration_hours, freq='h')
            
            # Sum up predicted carbon intensity for the task duration at this specific time
            try:
                predicted_cost = self.forecast.loc[execution_window].sum()
                
                if predicted_cost < lowest_carbon_cost:
                    lowest_carbon_cost = predicted_cost
                    best_start_time = slot
            except KeyError:
                # Handle edge cases where window goes out of forecast bounds
                continue
                
        # Calculate baseline cost (if we ran the task instantly upon arrival)
        try:
            baseline_window = pd.date_range(start=arrival_time, periods=duration_hours, freq='h')
            baseline_cost = self.forecast.loc[baseline_window].sum()
        except KeyError:
            baseline_cost = lowest_carbon_cost
            
        carbon_saved = baseline_cost - lowest_carbon_cost
        
        return {
            'task_id': task_id,
            'arrival_time': arrival_time,
            'sla_deadline': sla_deadline_time,
            'scheduled_start': best_start_time,
            'delay_hours': int((best_start_time - arrival_time).total_seconds() / 3600),
            'carbon_saved_units': round(carbon_saved, 2)
        }