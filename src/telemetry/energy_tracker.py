import os
import mlflow
from codecarbon import EmissionsTracker

class GreenTracker:
    def __init__(self, project_name="green_mlops_baseline"):
        # Ensure the logs directory exists
        os.makedirs("./logs", exist_ok=True)
        
        # Initialize the CodeCarbon tracker
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir="./logs",
            output_file="emissions.csv",
            log_level="error",
            measure_power_secs=1
        )
        

    def __enter__(self):
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        emissions = self.tracker.stop()
        self.final_emissions = emissions  # kg CO2e, exposed for callers that need the number
        print(f"\n[GreenOps Telemetry] Task Finished.")
        print(f"Total Carbon Footprint: {emissions:.6f} kg CO2e\n")

        if mlflow.active_run(): 
            mlflow.log_metric("carbon_footprint_kg_CO2e", emissions)
        # Automatically send the final carbon score to the active MLflow dashboard