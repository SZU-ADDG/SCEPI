import json
from pathlib import Path

from download_data import download

ROOT = Path(__file__).resolve().parents[1]
job = {
    "source": "University of Copenhagen ERDA", "record": "c2EWrbGSCV",
    "file": "predictions.zip", "bytes": 36080539,
    "path": "data/raw/kermut/predictions.zip",
    "url": "https://sid.erda.dk/share_redirect/c2EWrbGSCV/predictions.zip",
}
result = download(job)
(ROOT / "data/raw/kermut/predictions_manifest.json").write_text(json.dumps({"complete": True, "files": [result]}, indent=2))
