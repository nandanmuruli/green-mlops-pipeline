import time
import torch
import torch.nn as nn
from src.telemetry.energy_tracker import GreenTracker

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple linear network to generate some system load
        self.fc = nn.Sequential(
            nn.Linear(1000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1000)
        )

    def forward(self, x):
        return self.fc(x)

def test_telemetry():
    print("Initializing dummy PyTorch model...")
    model = DummyModel()
    dummy_input = torch.randn(256, 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Starting energy-tracked execution loop...")
    
    # Wrap the compute loop with our energy tracker
    with GreenTracker(project_name="telemetry_test_run"):
        for epoch in range(15):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_input)
            loss.backward()
            optimizer.step()
            time.sleep(0.5)
            
    print("Execution loop complete. Check your logs directory!")

if __name__ == "__main__":
    test_telemetry()