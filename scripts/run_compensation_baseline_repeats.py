import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/compensation_main"
CONFIGURATIONS = [
    ("full_g0_f0_j1_n1", ["--joint", "--no-interaction"], False),
    ("full_g0_f1_j1_n1", ["--flexible", "--joint", "--no-interaction"], True),
]

OUTPUT.mkdir(parents=True, exist_ok=True)
status_path = OUTPUT / "source_baseline_repeats.json"
completed = []

for tag, options, flexible in CONFIGURATIONS:
    grid_path = ROOT / "runs/e3" / f"pair_grid_{tag}" / "manifest.json"
    grid = json.loads(grid_path.read_text())
    assert grid["complete"] and not grid.get("smoke", False)
    assert grid["trials"] == 120
    config = grid["config"]
    assert set(config["thresholds"]) == {25, 30}
    assert config["representation"] == "full"
    assert config["geometry"] is False
    assert config["flexible"] is flexible
    assert config["joint"] is True
    assert config["no_interaction"] is True

    repeat_directory = ROOT / "runs/e3" / f"pair_repeats_{tag}"
    repeat_manifest = repeat_directory / "manifest.json"
    selection_path = repeat_directory / "selected_hyperparameters.json"
    if repeat_manifest.exists():
        repeats = json.loads(repeat_manifest.read_text())
        assert repeats["complete"] and not repeats.get("smoke", False)
        selected = json.loads(selection_path.read_text())
        assert set(selected) == {f"family{threshold}/fold{fold}" for threshold in [25, 30] for fold in range(5)}
        completed.append(tag)
        continue

    status_path.write_text(json.dumps({"state": "waiting_or_running", "stage": tag,
                                       "completed_stages": completed}, indent=2))
    command = [
        sys.executable, "scripts/run_when_capacity.py", "--memory-gib", "60", "--gpu-memory-gib", "2",
        "--status", f"runs/compensation_main/baseline_repeats_{tag}_capacity.json",
        sys.executable, "scripts/run_family_repeats.py", "--stage", "pair", "--thresholds", "30", "25",
        *options,
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    repeats = json.loads(repeat_manifest.read_text())
    assert repeats["complete"] and not repeats.get("smoke", False)
    selected = json.loads(selection_path.read_text())
    assert set(selected) == {f"family{threshold}/fold{fold}" for threshold in [25, 30] for fold in range(5)}
    completed.append(tag)
    status_path.write_text(json.dumps({"state": "running", "completed_stages": completed}, indent=2))

status_path.write_text(json.dumps({"complete": True, "completed_stages": completed}, indent=2))
