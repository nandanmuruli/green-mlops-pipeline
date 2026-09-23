"""Fast smoke tests for CI: verify config integrity and that the telemetry
wrapper can be constructed, without running a full training job."""
import os

from src.telemetry.energy_tracker import GreenTracker
from src.utils.config import load_config


def test_config_loads_and_has_required_sections():
    cfg = load_config()
    for section in ("dataset", "model", "training", "mlflow", "telemetry", "compression", "rq3"):
        assert section in cfg, f"missing config section: {section}"


def test_config_paths_are_relative_and_consistent():
    cfg = load_config()
    assert cfg["compression"]["baseline_checkpoint"].startswith("./models/")
    assert cfg["rq3"]["sla_latency_ms"] > 0
    assert len(cfg["rq3"]["thread_caps"]) > 0


def test_green_tracker_can_be_constructed(tmp_path):
    # Doesn't start the tracker (no I/O / no training) -- just verifies the
    # wrapper class initializes cleanly, which is what CI can cheaply check.
    os.chdir(tmp_path)
    tracker = GreenTracker(project_name="ci_smoke_test")
    assert tracker.tracker is not None
