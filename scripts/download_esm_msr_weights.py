import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_url

from download_data import download

ROOT = Path(__file__).resolve().parents[1]
sources = json.loads((ROOT / "work/sources/esm_msr_weights.json").read_text())
jobs = []
for repo, info in sources.items():
    folder = "esm3-sm-open-v1" if repo.startswith("biohub/") else "esm-msr"
    for item in info["files"]:
        filename = item["name"]
        if filename.startswith(".") or filename.endswith("1utn.pdb"):
            continue
        if "decoder_v0.pth" in filename:
            continue
        jobs.append({
            "source": "huggingface", "repo": repo, "record": info["revision"], "file": filename,
            "bytes": item["size"], "path": f"models/{folder}/{filename}",
            "url": hf_hub_url(repo, filename, revision=info["revision"]),
        })
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(download, jobs))
(ROOT / "models/esm_msr_download_manifest.json").write_text(json.dumps({"sources": sources, "files": results}, indent=2))
print(json.dumps({"complete": True, "files": len(results)}), flush=True)
