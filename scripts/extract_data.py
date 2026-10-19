import argparse
import json
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(path):
    receipt = path.with_name(path.name + ".extracted.json")
    if receipt.exists():
        print(receipt.read_text(), flush=True)
        return
    destination = ROOT / "data" / "extracted" / path.parent.name
    destination.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, str(path)
            members = [x for x in archive.infolist() if "__MACOSX" not in Path(x.filename).parts]
            archive.extractall(destination, members=members)
            total = sum(x.file_size for x in members)
            count = sum(not x.is_dir() for x in members)
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            archive.extractall(destination, members=members, filter="data")
            total = sum(x.size for x in members if x.isfile())
            count = sum(x.isfile() for x in members)
    else:
        raise ValueError(path)
    result = {"source": str(path.relative_to(ROOT)), "destination": str(destination.relative_to(ROOT)), "files": count, "uncompressed_bytes": total}
    receipt.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for name in args.paths:
        extract(ROOT / name)
