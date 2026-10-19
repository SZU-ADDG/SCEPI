import argparse
import json
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    result = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            members = [x for x in archive.infolist() if not x.is_dir()]
            result["members"] = [{"name": x.filename, "bytes": x.file_size} for x in members]
            result["uncompressed_bytes"] = sum(x.file_size for x in members)
            for member in members:
                if member.filename.endswith(".csv") and "__MACOSX" not in Path(member.filename).parts:
                    with archive.open(member) as stream:
                        frame = pd.read_csv(stream, nrows=2, low_memory=False)
                    result.setdefault("csv_samples", {})[member.filename] = json.loads(frame.to_json(orient="records"))
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            members = [x for x in archive.getmembers() if x.isfile()]
            result["members"] = [{"name": x.name, "bytes": x.size} for x in members]
            result["uncompressed_bytes"] = sum(x.size for x in members)
            for member in members:
                if member.name.endswith(".csv"):
                    with archive.extractfile(member) as stream:
                        frame = pd.read_csv(stream, nrows=2, low_memory=False)
                    result.setdefault("csv_samples", {})[member.name] = json.loads(frame.to_json(orient="records"))
    else:
        raise ValueError(path)
    output = ROOT / "data" / "processed" / "archive_inspection"
    output.mkdir(parents=True, exist_ok=True)
    (output / (path.name + ".json")).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["path", "bytes", "uncompressed_bytes"]}), flush=True)
    for name, rows in result.get("csv_samples", {}).items():
        print(json.dumps({"csv": name, "columns": list(rows[0]), "sample": rows[0]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        inspect(ROOT / path)


if __name__ == "__main__":
    main()
