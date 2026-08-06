import os
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
        print(f"\n[GreenOps Telemetry] Task Finished.")
        print(f"Total Carbon Footprint: {emissions:.6f} kg CO2e\n")