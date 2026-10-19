import json
from pathlib import Path
import zipfile

import fsspec

ROOT = Path(__file__).resolve().parents[1]
metadata = json.loads((ROOT / "work/sources/trpb_original.json").read_text())
entry = metadata["files"]["entries"]["data.zip"]
with fsspec.open(entry["links"]["content"], mode="rb", block_size=1048576, client_kwargs={"trust_env": True}) as stream:
    with zipfile.ZipFile(stream) as archive:
        members = [{"name": item.filename, "bytes": item.file_size} for item in archive.infolist() if not item.is_dir()]
(ROOT / "work/sources/trpb_data_members.json").write_text(json.dumps(members, indent=2))
for item in members:
    if any(word in item["name"].lower() for word in ["fasta", "reference", ".fa", "readme", "wt"]):
        print(json.dumps(item), flush=True)
print(json.dumps({"members": len(members)}), flush=True)
