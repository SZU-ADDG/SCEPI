import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_url, model_info

from download_data import download

ROOT = Path(__file__).resolve().parents[1]
REPO = "facebook/esm2_t33_650M_UR50D"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"
destination = ROOT / "models" / "esm2_t33_650M_UR50D"
info = model_info(REPO, revision=REVISION, files_metadata=True)
assert info.sha == REVISION
partial = destination / "model.safetensors.part"
incomplete = list((destination / "cache/huggingface/download").glob("*.incomplete"))
if incomplete:
    assert len(incomplete) == 1 and not partial.exists(), incomplete
    os.replace(incomplete[0], partial)
results = []
for entry in info.siblings:
    if not (entry.rfilename.endswith(".json") or entry.rfilename in {"README.md", "vocab.txt", "model.safetensors"}):
        continue
    results.append(download({
        "source": "huggingface", "record": REVISION, "repo": REPO,
        "file": entry.rfilename, "bytes": entry.size,
        "path": str((destination / entry.rfilename).relative_to(ROOT)),
        "url": hf_hub_url(REPO, entry.rfilename, revision=REVISION),
    }))
manifest = {"repo": REPO, "revision": REVISION, "path": str(destination.relative_to(ROOT)), "files": results}
(destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"repo": REPO, "revision": REVISION, "files": len(results)}), flush=True)
