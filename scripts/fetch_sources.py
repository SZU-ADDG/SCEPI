import concurrent.futures
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = [
    ("ThermoMPNN-D", "Kuhlman-Lab/ThermoMPNN-D", "df9a75aaddb674a7c4c193005031fc0536d325fb", "main"),
    ("kermut", "petergroth/kermut", "e6500221b5159a1896272bfc4a3c6e3896eb0027", "reproduce"),
    ("ProteinGym", "OATML-Markslab/ProteinGym", "144fe22b07dfaeec2b366f2346203a9838a55b4c", "main"),
    ("esm-msr", "SKTeamLab/esm-msr", "6ecea374ecf2c661065449227a0e93cb825d5162", "main"),
    ("pymochi", "lehner-lab/MoCHI", "ad6c1e152799b5c323c80dcac99872fe4d4fbb16", "main"),
    ("esm", "facebookresearch/esm", "2b369911bb5b4b0dda914521b9475cad1656b2ac", "main"),
    ("RFA", "JoeThorntonLab/RFA", "7a1a8725315a027d9563f6d9b12749174cffefc0", "main"),
]


def fetch(item):
    name, repository, revision, branch = item
    destination = ROOT / "external" / name
    url = f"https://github.com/{repository}.git"
    if not destination.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
    observed = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    assert observed == revision, (name, observed, revision)
    return {"name": name, "url": url, "revision": revision, "path": str(destination.relative_to(ROOT))}


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(fetch, REPOSITORIES))
    (ROOT / "external" / "sources.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
