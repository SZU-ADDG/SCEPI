import json
from pathlib import Path
import subprocess

from Bio import SearchIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/external_transfer"
PFAM = ROOT / "data/processed/pfam37"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = []
    for dataset in ["functional_doubles", "flip_primary"]:
        table = pd.read_parquet(ROOT / "data/processed" / dataset / "proteins.parquet")
        table["dataset"] = dataset
        tables.append(table[["protein_id", "reference_sequence", "dataset"]])
    proteins = pd.concat(tables, ignore_index=True)
    assert proteins.protein_id.is_unique
    fasta = OUTPUT / "references.fasta"
    SeqIO.write([SeqRecord(Seq(row.reference_sequence), id=row.protein_id, description=row.dataset) for row in proteins.itertuples()],
                fasta, "fasta")
    alignment_path = OUTPUT / "source_alignments.tsv"
    alignment_command = [
        "mmseqs", "easy-search", str(fasta), str(ROOT / "data/processed/tsuboyama/references.fasta"),
        str(alignment_path), str(ROOT / "work/mmseqs_external"), "--min-seq-id", "0.25",
        "-c", "0.8", "--cov-mode", "0", "--prefilter-mode", "2", "--alignment-mode", "3",
        "--max-seqs", "10000", "-e", "100000", "--threads", "16",
        "--format-output", "query,target,fident,qcov,tcov,alnlen,qstart,qend,tstart,tend,evalue,bits",
    ]
    if not alignment_path.exists():
        with (OUTPUT / "mmseqs.log").open("w") as handle:
            subprocess.run(alignment_command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    domains = OUTPUT / "external.domtbl"
    hmmer_command = ["hmmscan", "--cpu", "16", "--cut_ga", "--noali", "--domtblout", str(domains),
                     "-o", str(OUTPUT / "external.hmmscan"), str(PFAM / "Pfam-A.hmm"), str(fasta)]
    if not domains.exists():
        with (OUTPUT / "hmmer.log").open("w") as handle:
            subprocess.run(hmmer_command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    alignments = pd.read_csv(alignment_path, sep="\t", names=[
        "query", "target", "identity", "qcov", "tcov", "alnlen", "qstart", "qend", "tstart", "tend", "evalue", "bits"])
    clans = pd.read_csv(ROOT / "data/raw/pfam37/Pfam-A.clans.tsv.gz", sep="\t", header=None,
                        names=["pfam", "clan", "clan_name", "pfam_name", "description"]).set_index("pfam")
    annotations = []
    for query in SearchIO.parse(domains, "hmmscan3-domtab"):
        for hit in query:
            accession = hit.accession.split(".")[0]
            assert accession in clans.index
            for hsp in hit.hsps:
                annotations.append({"protein_id": query.id, "pfam": accession, "clan": clans.loc[accession, "clan"],
                                    "evalue": hsp.evalue, "bitscore": hsp.bitscore,
                                    "sequence_start": hsp.query_start, "sequence_end": hsp.query_end})
    annotations = pd.DataFrame(annotations)
    assert len(annotations) > 0
    annotations.to_parquet(OUTPUT / "annotations.parquet", index=False)
    source_annotations = pd.read_parquet(PFAM / "tsuboyama_annotations.parquet")
    all_permissions = []
    for threshold in [30, 25]:
        families = pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet")
        edges = []
        selected = alignments.loc[alignments.identity.ge(threshold / 100) & alignments.qcov.ge(0.8) & alignments.tcov.ge(0.8)]
        for row in selected.itertuples():
            edges.append({"protein_id": row.query, "source_protein_id": row.target, "reason": "sequence_identity_coverage"})
        for field in ["pfam", "clan"]:
            joined = annotations.dropna(subset=[field]).merge(source_annotations.dropna(subset=[field]), on=field, suffixes=("", "_source"))
            for row in joined.itertuples():
                edges.append({"protein_id": row.protein_id, "source_protein_id": row.protein_id_source,
                              "reason": field + ":" + getattr(row, field)})
        edges = pd.DataFrame(edges).drop_duplicates()
        edges.to_parquet(OUTPUT / f"family_matches_{threshold}.parquet", index=False)
        for protein in proteins.itertuples():
            matched = set(edges.loc[edges.protein_id.eq(protein.protein_id), "source_protein_id"])
            source_families = set(families.loc[families.protein_id.isin(matched), "family_group"])
            related = families.loc[families.family_group.isin(source_families)]
            folds = sorted(related.outer_group.unique().tolist())
            eligible = [fold for fold in range(5) if all(value == fold for value in folds)]
            all_permissions.append({
                "protein_id": protein.protein_id, "dataset": protein.dataset, "threshold": threshold,
                "related_source_proteins": sorted(related.protein_id.tolist()),
                "related_source_families": sorted(source_families), "related_source_outer_groups": folds,
                "eligible_source_test_folds": eligible, "requires_custom_source_exclusion": len(eligible) == 0,
                "selection_rule": "Related source families are absent from both training and hyperparameter validation.",
            })
    pd.DataFrame(all_permissions).to_parquet(OUTPUT / "source_permissions.parquet", index=False)
    result = {
        "complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "proteins": len(proteins), "thresholds": [30, 25], "pfam": "37.0",
        "alignment_command": alignment_command, "hmmer_command": hmmer_command,
        "existing_source_folds_modified": False, "target_labels_used": 0, "permissions": all_permissions,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(result, indent=2))
    print(pd.DataFrame(all_permissions)[["protein_id", "dataset", "threshold", "related_source_outer_groups", "eligible_source_test_folds"]].to_string(index=False))


if __name__ == "__main__":
    main()
