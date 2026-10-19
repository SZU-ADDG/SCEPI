import json
from pathlib import Path
import subprocess

import requests

from download_data import download

ROOT = Path(__file__).resolve().parents[1]
metadata = json.loads((ROOT / "work/sources/pdz_original.json").read_text())
assert len(metadata["files"]) == 1
entry = metadata["files"][0]
record = download({"source": "PDZ3 original measurements", "record": str(metadata["id"]),
                   "file": entry["key"], "bytes": entry["size"], "url": entry["links"]["self"],
                   "path": "data/raw/flip_original_pdz/" + entry["key"]})
url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs42003-021-01878-9/MediaObjects/42003_2021_1878_MOESM3_ESM.xlsx"
path = ROOT / "data/raw/flip_original_rhomax/42003_2021_1878_MOESM3_ESM.xlsx"
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
assert path.stat().st_size > 0
repository = ROOT / "external/nuclease_design"
nucb = repository / "data/landscape.csv"
assert nucb.is_file()
manifest = {"complete": True, "source_commit": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "pdz3": record,
            "rhomax": {"url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size},
            "nucb": {"repository": "https://github.com/google-deepmind/nuclease_design",
                     "revision": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
                     "path": str(nucb.relative_to(ROOT)), "bytes": nucb.stat().st_size}}
(ROOT / "data/raw/flip_supplementary_original_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
