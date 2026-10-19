import json
from pathlib import Path
from download_data import download

ROOT = Path(__file__).resolve().parents[1]
job = {
    "source": "University of Copenhagen ERDA", "record": "c2EWrbGSCV",
    "file": "kermut_data.zip", "bytes": 4062060895,
    "path": "data/raw/kermut/kermut_data.zip",
    "url": "https://sid.erda.dk/share_redirect/c2EWrbGSCV/kermut_data.zip",
}
result = download(job)
(ROOT / "data/raw/kermut/manifest.json").write_text(json.dumps({"complete": True, "files": [result]}, indent=2))
