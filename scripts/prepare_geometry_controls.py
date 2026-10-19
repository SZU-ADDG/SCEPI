import json
from pathlib import Path
import subprocess

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / "data/features/training/tsuboyama"
    table = pd.read_parquet(root / "doubles/metadata.parquet")
    original_distance = np.load(root / "doubles/ca_distance.npy")
    original_confidence = np.load(root / "doubles/confidence.npy")
    values = {kind: {"ca_distance": original_distance.copy(), "confidence": original_confidence.copy()}
              for kind in ["permuted", "rewired"]}
    records = {kind: [] for kind in values}
    for protein_index, (protein, group) in enumerate(table.loc[table.structure_available].groupby("protein_id", sort=True)):
        geometry = np.load(ROOT / "data/features/structures/tsuboyama" / f"{protein}.npz")
        coordinates, confidence = geometry["coordinates"], geometry["confidence"]
        first, second = np.triu_indices(len(coordinates), 1)
        distances = pdist(coordinates).astype(np.float32)
        pair_confidence = confidence[np.stack([first, second], axis=1)]
        index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(first, second))}
        selected = np.array([index[tuple(pair)] for pair in group.positions])
        np.testing.assert_allclose(distances[selected], original_distance[group.matrix_row], atol=1e-5, rtol=1e-5)
        gaps = second - first
        bins = np.digitize(gaps, [5, 10, 20, 40])
        rng = np.random.default_rng(20261009 + protein_index)
        permutation = np.arange(len(first))
        for group_id in np.unique(bins):
            members = np.flatnonzero(bins == group_id)
            permutation[members] = rng.permutation(members)
            np.testing.assert_array_equal(np.sort(distances[permutation[members]]), np.sort(distances[members]))
        permuted_distance, permuted_confidence = distances[permutation], pair_confidence[permutation]
        values["permuted"]["ca_distance"][group.matrix_row] = permuted_distance[selected]
        values["permuted"]["confidence"][group.matrix_row] = permuted_confidence[selected]
        contact_edges = {(int(first[i]), int(second[i])) for i in np.flatnonzero(distances <= 8)}
        graph = nx.Graph()
        graph.add_nodes_from(range(len(coordinates)))
        graph.add_edges_from(sorted(contact_edges))
        degrees = dict(graph.degree())
        assert graph.number_of_edges() >= 2 and graph.number_of_nodes() >= 4
        nx.double_edge_swap(graph, nswap=10 * graph.number_of_edges(), max_tries=1000 * graph.number_of_edges(),
                            seed=20261009 + protein_index)
        assert dict(graph.degree()) == degrees
        changed_edges = {tuple(sorted(edge)) for edge in graph.edges()}
        removed, added = sorted(contact_edges - changed_edges), sorted(changed_edges - contact_edges)
        assert len(removed) == len(added)
        rewired = distances.copy()
        for old, new in zip(removed, added):
            a, b = index[old], index[new]
            rewired[a], rewired[b] = distances[b], distances[a]
        np.testing.assert_array_equal(np.sort(rewired), np.sort(distances))
        observed_edges = {(int(first[i]), int(second[i])) for i in np.flatnonzero(rewired <= 8)}
        assert observed_edges == changed_edges
        values["rewired"]["ca_distance"][group.matrix_row] = rewired[selected]
        ledger = pd.DataFrame({"protein_id": protein, "position_i": first, "position_j": second,
                               "sequence_gap_bin": bins, "original_distance": distances,
                               "permuted_from_i": first[permutation], "permuted_from_j": second[permutation],
                               "permuted_distance": permuted_distance, "rewired_distance": rewired})
        destination = root / "geometry_controls" / protein
        destination.mkdir(parents=True, exist_ok=True)
        ledger.to_parquet(destination / "all_position_pairs.parquet", index=False)
        for kind, altered in [("permuted", permuted_distance), ("rewired", rewired)]:
            records[kind].append({"protein_id": protein, "all_position_pairs": len(first),
                                  "observed_variants": len(group), "observed_site_pairs": group.pair_id.nunique(),
                                  "changed_position_pairs": int(np.count_nonzero(altered != distances)),
                                  "changed_observed_variants": int(np.count_nonzero(altered[selected] != distances[selected])),
                                  "original_contact_edges": len(contact_edges), "exchanged_contact_edges": len(removed)})
    for kind, arrays in values.items():
        output = root / ("geometry_" + kind)
        output.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            assert np.isfinite(array).all()
            np.save(output / f"{name}.npy", array)
        pd.DataFrame(records[kind]).to_csv(output / "coverage.csv", index=False)
        manifest = {"complete": True, "control": kind, "seed": 20261009,
                    "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                    "proteins": len(records[kind]), "rows": len(table), "experimental_labels_used": 0,
                    "permutation_population": "All unordered residue-position pairs within each reference protein.",
                    "sequence_gap_bin_boundaries": [5, 10, 20, 40],
                    "permuted_fields": ["distance", "both_endpoint_confidences"] if kind == "permuted" else ["distance"],
                    "rewiring": "Degree-preserving double-edge swaps at 8 Angstrom; swap complete distance values between removed and added contacts.",
                    "ledger": str((root / "geometry_controls").relative_to(ROOT)),
                    "checks": ["Exact feature-row mapping", "Within-bin distance multiset conservation",
                               "Contact degree conservation", "Global distance multiset conservation"]}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
