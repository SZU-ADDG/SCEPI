import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
records = []
for jobs_path in sorted((ROOT / "runs").glob("*/**/jobs.json")):
    jobs = json.loads(jobs_path.read_text())
    if not isinstance(jobs, list) or not jobs:
        continue
    completed, missing = 0, 0
    for job in jobs:
        if "stages" in job:
            paths = [ROOT / stage["manifest"] for stage in job["stages"]]
        elif "run" in job:
            paths = [ROOT / job["run"] / "manifest.json"]
        else:
            continue
        finished = all(path.exists() and json.loads(path.read_text()).get("complete", False) for path in paths)
        completed += int(finished)
        missing += int(not finished)
    if completed + missing:
        records.append({"group": str(jobs_path.parent.relative_to(ROOT)),
                        "expected_jobs": completed + missing, "completed_jobs": completed, "remaining_jobs": missing})
targets = []
for name in ["shared_target", "shared_target_only", "shared_target_geometry", "shared_target_only_geometry",
             "functional_adaptation", "mochi", "mavenn", "rfa", "mean_ridge", "mean_mlp", "kermut_target",
             "thermo_target", "mean_family_target", "mean_flip", "msr_target", "flip_supplementary"]:
    directory = ROOT / "runs" / name
    counts = {}
    for path in sorted(directory.rglob("manifest.json")):
        item = json.loads(path.read_text())
        if "evaluated" not in item:
            continue
        dataset = item.get("dataset", "tsuboyama")
        key = (dataset, item.get("method", item.get("supervision", name)))
        count = counts.setdefault(key, {"completed_fits": 0, "recorded_ineligible": 0})
        assert item["complete"]
        count["completed_fits" if item["evaluated"] else "recorded_ineligible"] += 1
    for (dataset, method), count in sorted(counts.items()):
        targets.append({"run_root": name, "dataset": dataset, "method": method, **count})
original = {}
for scope in ["domain", "multiples", "multiples_exact", "singles", "cpu"]:
    directory = ROOT / "runs/e1_kermut" / scope
    paths = list(directory.glob("*/*/manifest.json"))
    original["kermut_" + scope] = {"completed_assay_protocols": sum(json.loads(path.read_text()).get("complete", False) for path in paths),
                                  "completed_workers": len(list(directory.glob("manifest_rank*.json")))}
data = {}
for name, path in {
    "tsuboyama": "data/processed/tsuboyama/manifest.json",
    "functional": "data/processed/functional/manifest.json",
    "flip_primary": "data/processed/flip_primary/manifest.json",
    "flip_supplementary": "data/processed/flip_supplementary/manifest.json",
    "supplementary_original_verification": "runs/flip_supplementary_data_verification/manifest.json",
    "supplementary_features": "data/features/mean/flip_supplementary/manifest.json",
    "physical_mean_features": "data/features/mean/tsuboyama/manifest.json",
    "functional_mean_features": "data/features/mean/functional_doubles/manifest.json",
    "flip_mean_features": "data/features/mean/flip_primary/manifest.json",
    "msr_index": "data/features/esm_msr/tsuboyama/index_manifest.json",
}.items():
    source = ROOT / path
    data[name] = {"manifest": path, "present": source.exists()}
    if source.exists():
        info = json.loads(source.read_text())
        data[name].update({key: info[key] for key in ["complete", "variants", "proteins", "rows", "training_eligible",
                                                    "inputs", "records_verified", "encoded_sequences"] if key in info})
snapshot = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "project_directory": str(ROOT), "conda_environment": "/data1/doomx/anaconda3/envs/recomb-3-epistasis",
    "all_experiments_complete": False,
    "data": data, "scheduled_source_groups": records, "target_fits": targets, "original_kermut": original,
    "gpu_snapshot": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                                             "--format=csv"], text=True).strip(),
    "remaining_protocol_work": [
        "Finish every admitted source search, five-repeat evaluation, target adaptation and pending baseline reproduction.",
        "Complete ESM-MSR source training and evaluation with current family partitions.",
        "Complete mean-embedding and structural baseline S/C adaptation under matched label permissions.",
        "Complete the physical target-only shared-model C comparison.",
        "Complete applicable FLIP target-only baselines and NucB, PDZ and Rhomax supplementary evaluation.",
        "Complete all mechanism controls, geometry interventions, measurement/structure strata and fixed statistical tests.",
        "Reconcile all final predictions, metrics, figures and experiment manifests against the complete protocol."
    ]
}
output = ROOT / "runs/progress"
output.mkdir(parents=True, exist_ok=True)
(output / "current.json").write_text(json.dumps(snapshot, indent=2))
print(json.dumps(snapshot, indent=2))
