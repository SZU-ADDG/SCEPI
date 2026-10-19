import concurrent.futures
import json
from pathlib import Path

import requests

from download_data import download

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/"
FILES = ["Pfam-A.hmm.gz", "Pfam-A.clans.tsv.gz", "Pfam.version.gz", "relnotes.txt"]


def fetch(name):
    url = BASE + name
    response = requests.head(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    return download({
        "source": "Pfam", "record": "37.0", "file": name,
        "bytes": int(response.headers["Content-Length"]),
        "path": str(Path("data/raw/pfam37") / name), "url": url,
    })


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(fetch, FILES))
    destination = ROOT / "data/raw/pfam37/download_manifest.json"
    destination.write_text(json.dumps({"release": "37.0", "source": BASE, "files": files}, indent=2))
