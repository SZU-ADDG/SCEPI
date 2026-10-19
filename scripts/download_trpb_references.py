import json
from pathlib import Path
import zipfile

import fsspec

ROOT = Path(__file__).resolve().parents[1]
metadata = json.loads((ROOT / "work/sources/trpb_original.json").read_text())
entry = metadata["files"]["entries"]["data.zip"]
names = [
    "data/ftmlde_data/tm9d8s_AAs.fasta",
    "data/ftmlde_data/TARGET_b0.3.fa",
    "data/ftmlde_data/Tm9D8s_refine_19.pdb",
]
destination = ROOT / "data/raw/flip_original_trpb/reference_files"
destination.mkdir(parents=True, exist_ok=True)
records = []
with fsspec.open(entry["links"]["content"], mode="rb", block_size=1048576, client_kwargs={"trust_env": True}) as stream:
    with zipfile.ZipFile(stream) as archive:
        for name in names:
            content = archive.read(name)
            info = archive.getinfo(name)
            assert len(content) == info.file_size
            path = destination / Path(name).name
            path.write_bytes(content)
            records.append({"member": name, "path": str(path.relative_to(ROOT)), "bytes": len(content)})
manifest = {"complete": True, "record": "h5rah-5z170", "archive": "data.zip", "url": entry["links"]["content"], "files": records}
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
