import json
from pathlib import Path
import subprocess

from Bio import SearchIO
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/tsuboyama"
OUTPUT = ROOT / "data/splits"
FAMILY = ROOT / "data/processed/pfam37"
WORK = ROOT / "work/mmseqs_tsuboyama"
SEED = 20261009


def run_external():
    WORK.mkdir(parents=True, exist_ok=True)
    alignments = WORK / "all_pairs.tsv"
    fasta = DATA / "references.fasta"
    if not alignments.exists():
        command = [
            "mmseqs", "easy-search", str(fasta), str(fasta), str(alignments), str(WORK / "work"),
            "--min-seq-id", "0.25", "-c", "0.8", "--cov-mode", "0",
            "--prefilter-mode", "2", "--alignment-mode", "3", "--max-seqs", "10000",
            "-e", "100000", "--threads", "16",
            "--format-output", "query,target,fident,qcov,tcov,alnlen,qstart,qend,tstart,tend,evalue,bits",
        ]
        with (ROOT / "runs/mmseqs_tsuboyama.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        (WORK / "command.json").write_text(json.dumps(command, indent=2))
    domains = FAMILY / "tsuboyama.domtbl"
    if not domains.exists():
        command = [
            "hmmscan", "--cpu", "16", "--cut_ga", "--noali", "--domtblout", str(domains),
            "-o", str(FAMILY / "tsuboyama.hmmscan"), str(FAMILY / "Pfam-A.hmm"), str(fasta),
        ]
        with (ROOT / "runs/pfam_tsuboyama.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        (FAMILY / "command.json").write_text(json.dumps(command, indent=2))
    return alignments, domains


def connect_group(graph, values, provenance, edges):
    values = sorted(set(values))
    for value in values[1:]:
        graph.add_edge(values[0], value)
        edges.append({"query": values[0], "target": value, "reason": provenance})


def assign_outer_groups(table, counts):
    rng = np.random.default_rng(SEED)
    grouped = []
    for group, frame in table.groupby("family_group"):
        total = int(sum(counts.get(protein, 0) for protein in frame.protein_id))
        grouped.append((group, total, len(frame), rng.random()))
    grouped.sort(key=lambda item: (-item[1], -item[2], item[3]))
    loads = np.zeros((5, 2), dtype=np.int64)
    assignment = {}
    for group, total, size, _ in grouped:
        fold = min(range(5), key=lambda index: (loads[index, 0], loads[index, 1], index))
        assignment[group] = fold
        loads[fold] += [total, size]
    return table.family_group.map(assignment), loads.tolist()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_parquet(DATA / "proteins.parquet")
    quartets = pd.read_parquet(DATA / "quartets.parquet")
    main_quartets = quartets.loc[quartets.main]
    counts = main_quartets.groupby("protein_id").size().to_dict()
    alignments, domains = run_external()
    matches = pd.read_csv(alignments, sep="\t", names=["query", "target", "identity", "qcov", "tcov", "alnlen", "qstart", "qend", "tstart", "tend", "evalue", "bits"])
    assert set(proteins.protein_id) == set(matches.loc[matches["query"].eq(matches.target), "query"])
    assert matches.identity.between(0, 1).all()
    clans = pd.read_csv(ROOT / "data/raw/pfam37/Pfam-A.clans.tsv.gz", sep="\t", header=None, names=["pfam", "clan", "clan_name", "pfam_name", "description"])
    clan_lookup = clans.set_index("pfam")
    annotations = []
    for query in SearchIO.parse(domains, "hmmscan3-domtab"):
        for hit in query:
            accession = hit.accession.split(".")[0]
            assert accession in clan_lookup.index
            for hsp in hit.hsps:
                annotations.append({
                    "protein_id": query.id, "pfam": accession, "pfam_name": hit.id,
                    "clan": clan_lookup.loc[accession, "clan"], "evalue": hsp.evalue, "bitscore": hsp.bitscore,
                    "sequence_start": hsp.query_start, "sequence_end": hsp.query_end,
                    "model_start": hsp.hit_start, "model_end": hsp.hit_end,
                })
    annotations = pd.DataFrame(annotations)
    assert len(annotations) > 0
    annotations.to_parquet(FAMILY / "tsuboyama_annotations.parquet", index=False)
    results = {}
    for threshold in (0.30, 0.25):
        graph = nx.Graph()
        graph.add_nodes_from(proteins.protein_id)
        selected = matches.loc[
            matches.identity.ge(threshold) & matches.qcov.ge(0.80) & matches.tcov.ge(0.80)
            & ~matches["query"].eq(matches.target)
        ]
        graph.add_edges_from(zip(selected["query"], selected.target))
        homology = {}
        for index, component in enumerate(sorted(nx.connected_components(graph), key=lambda nodes: min(nodes))):
            homology.update({node: f"homology_{index:04d}" for node in component})
        edges = [{"query": row.query, "target": row.target, "reason": "sequence_identity_coverage"} for row in selected.itertuples()]
        for scaffold, frame in proteins.groupby("source_scaffold"):
            connect_group(graph, frame.protein_id, "reference_scaffold:" + scaffold, edges)
        for field in ("pfam", "clan"):
            for annotation, frame in annotations.dropna(subset=[field]).groupby(field):
                connect_group(graph, frame.protein_id, field + ":" + annotation, edges)
        families = {}
        for index, component in enumerate(sorted(nx.connected_components(graph), key=lambda nodes: min(nodes))):
            families.update({node: f"group_{index:04d}" for node in component})
        table = proteins[["protein_id", "source_reference", "source_scaffold", "origin"]].copy()
        table["homology_group"] = table.protein_id.map(homology)
        table["family_group"] = table.protein_id.map(families)
        table["pfam_annotated"] = table.protein_id.isin(annotations.protein_id)
        table["outer_group"], loads = assign_outer_groups(table, counts)
        table["main_quartets"] = table.protein_id.map(counts).fillna(0).astype(int)
        fold = table.set_index("protein_id").outer_group
        assert all(fold.loc[row.query] == fold.loc[row.target] for row in selected.itertuples())
        assert all(fold.loc[row["query"]] == fold.loc[row["target"]] for row in edges)
        tag = f"{round(threshold * 100)}"
        table.to_parquet(OUTPUT / f"families_{tag}.parquet", index=False)
        pd.DataFrame(edges).to_parquet(OUTPUT / f"family_edges_{tag}.parquet", index=False)
        results[tag] = {
            "identity_threshold": threshold, "bidirectional_coverage": 0.8,
            "homology_groups": table.homology_group.nunique(), "family_groups": table.family_group.nunique(),
            "pfam_annotated_proteins": int(table.pfam_annotated.sum()),
            "pfam_annotated_groups_with_main_quartets": table.loc[table.pfam_annotated & table.main_quartets.gt(0)].family_group.nunique(),
            "outer_group_loads_quartets_proteins": loads, "cross_group_qualifying_alignments": 0,
            "path": str((OUTPUT / f"families_{tag}.parquet").relative_to(ROOT)),
        }
    result = {
        "stage": "E3_split", "seed": SEED, "pfam": "37.0", "hmmer": "3.4",
        "mmseqs_version": subprocess.check_output(["mmseqs", "version"], text=True).strip(),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "protocol": "Exhaustive pairwise search without a sequence prefilter; family/clan and documented reference scaffold merges.",
        "scope": "Pfam-annotated families and unannotated sequence homology groups are flagged separately.",
        "results": results,
    }
    (OUTPUT / "family_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
