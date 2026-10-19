import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "work" / "sources"
RAW = ROOT / "data" / "raw"
GROUPS = {
    "tsuboyama": ("tsuboyama_zenodo.json", {"Processed_K50_dG_datasets.zip", "AlphaFold_model_PDBs.zip", "Data_tables_for_figs.zip"}),
    "thermompnn": ("thermompnn_zenodo.json", {"Megascale.tar.gz", "PTMUL-D.tar.gz", "DMS.tar.gz", "Megascale-Rosetta-Sweeps.tar.gz"}),
    "esm_msr": ("esm_msr_zenodo.json", {"preprocessed_data.tar.gz", "all_model_predictions.tar.gz"}),
    "flip2": ("flip2_zenodo.json", None),
    "proteingym": ("proteingym_zenodo.json", {"DMS_ProteinGym_substitutions.zip", "substitutions_raw_DMS.zip", "ProteinGym_AF2_structures.zip", "cv_folds_singles_substitutions.zip", "cv_folds_multiples_substitutions.zip"}),
}
FLIP_PREFIXES = {"trpb", "hydro", "amylase", "ired", "nucb", "pdz3", "rhomax"}


def download(item):
    path = ROOT / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.stat().st_size == item["bytes"], str(path)
        return {**item, "state": "complete", "elapsed_seconds": 0}
    partial = path.with_name(path.name + ".part")
    start = time.monotonic()
    print(json.dumps({"state": "downloading", "path": item["path"], "bytes": item["bytes"]}), flush=True)
    subprocess.run([
        "aria2c", "--continue=true", "--auto-file-renaming=false",
        "--allow-overwrite=false", "--file-allocation=none",
        "--split=8", "--max-connection-per-server=8", "--min-split-size=4M",
        "--max-tries=12", "--retry-wait=5", "--timeout=60", "--connect-timeout=30",
        "--summary-interval=0", "--console-log-level=warn", "--download-result=hide",
        "--show-console-readout=false",
        "--all-proxy=" + os.environ["ALL_PROXY"],
        "--dir=" + str(partial.parent), "--out=" + partial.name, item["url"],
    ], check=True)
    assert partial.stat().st_size == item["bytes"], str(partial)
    os.replace(partial, path)
    result = {**item, "state": "complete", "elapsed_seconds": round(time.monotonic() - start, 2)}
    receipt = path.with_name(path.name + ".download.json")
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    args = parser.parse_args()
    suffix = "_" + "_".join(args.groups)
    items = []
    for group in args.groups:
        metadata_name, allowed = GROUPS[group]
        metadata = json.loads((SOURCES / metadata_name).read_text())
        selected = []
        for entry in metadata["files"]:
            key = entry["key"]
            if allowed is not None and key not in allowed:
                continue
            if group == "flip2" and key.split("/")[0] not in FLIP_PREFIXES:
                continue
            selected.append(key)
            items.append({
                "source": group,
                "record": str(metadata["id"]),
                "file": key,
                "bytes": entry["size"],
                "url": entry["links"]["self"],
                "path": str(Path("data/raw") / group / key),
            })
        if allowed is not None:
            assert set(selected) == allowed, (group, selected, allowed)
    plan = {"files": items, "total_bytes": sum(x["bytes"] for x in items)}
    (RAW / ("download_plan" + suffix + ".json")).write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(json.dumps({"files": len(items), "total_bytes": plan["total_bytes"]}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, items))
    (RAW / ("download_manifest" + suffix + ".json")).write_text(json.dumps({"files": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
