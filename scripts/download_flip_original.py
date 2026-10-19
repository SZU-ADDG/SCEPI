import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_data import download

ROOT = Path(__file__).resolve().parents[1]
jobs = []
for dataset in ["trpb", "ired", "hydro"]:
    metadata = json.loads((ROOT / "work/sources" / f"{dataset}_original.json").read_text())
    files = metadata["files"]["entries"].values() if dataset == "trpb" else metadata["files"]
    for entry in files:
        if dataset == "trpb" and entry["key"] != "code.zip":
            continue
        jobs.append({
            "source": "FLIP2 original source", "record": str(metadata["id"]), "file": entry["key"],
            "path": f"data/raw/flip_original_{dataset}/" + entry["key"], "bytes": entry["size"],
            "url": entry["links"]["content"] if dataset == "trpb" else entry["links"]["self"],
        })
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(download, jobs))
(ROOT / "data/raw/flip_original_manifest.json").write_text(json.dumps({"complete": True, "files": results}, indent=2))
