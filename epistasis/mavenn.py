import gc
import importlib.metadata
import json
from pathlib import Path

from Bio.Seq import MutableSeq
import mavenn
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf

from epistasis.metrics import correlation

ROOT = Path(__file__).resolve().parents[1]


class MavennRegression:
    runtime_device = None

    def __init__(self, features, dataset, order, device):
        self.features, self.order = features, order
        if MavennRegression.runtime_device is None:
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            devices = tf.config.list_physical_devices("GPU")
            if device == "cuda":
                assert devices, "TensorFlow GPU libraries are unavailable"
                for current in devices:
                    tf.config.experimental.set_memory_growth(current, True)
            else:
                tf.config.set_visible_devices([], "GPU")
            MavennRegression.runtime_device = device
        assert MavennRegression.runtime_device == device
        proteins = pd.read_parquet(ROOT / "data/processed" / dataset / "proteins.parquet").set_index("protein_id")
        protein = features.metadata.protein_id.unique()
        assert len(protein) == 1
        self.reference = proteins.loc[protein[0], "reference_sequence"]
        sequences = features.metadata.aa_seq.tolist() + [self.reference]
        for row in features.single_metadata.itertuples():
            sequence = MutableSeq(self.reference)
            assert self.reference[row.positions[0]] == row.wt_residues
            sequence[row.positions[0]] = row.mut_residues
            sequences.append(str(sequence))
        self.positions = sorted({int(row.positions[0]) for row in features.single_metadata.itertuples()})
        self.sequences = np.asarray(["".join(sequence[position] for position in self.positions) for sequence in sequences])
        _, first, inverse = np.unique(self.sequences, return_index=True, return_inverse=True)
        self.prediction_alias = first[inverse]
        self.alphabet = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
        encoder = OrdinalEncoder(categories=[self.alphabet] * len(self.positions), dtype=np.int16)
        self.encoded = encoder.fit_transform(np.asarray([list(sequence) for sequence in self.sequences]))
        self.reference_encoded = self.encoded[len(features.metadata)]
        self.batch_size = 50 if order == 2 else 512
        self.device = device

    def ohe(self, rows):
        return np.eye(20, dtype=np.float32)[self.encoded[rows]].reshape(len(rows), -1)

    def predict(self, model, rows):
        predictions = []
        for start in range(0, len(rows), self.batch_size):
            x = tf.convert_to_tensor(self.ohe(rows[start:start + self.batch_size]))
            latent = model.model.x_to_phi_layer(x)
            phenotype = model.model.phi_to_yhat_layer(latent)
            predictions.append(phenotype.numpy().ravel() * model.y_std + model.y_mean)
        result = np.concatenate(predictions)
        assert np.isfinite(result).all()
        return result

    def fit(self, train_indices, validation_indices, points, seed, directory, epsilon_components=None):
        assert not set(train_indices) & set(validation_indices)
        assert np.isfinite(points[train_indices]).all() and np.isfinite(points[validation_indices]).all()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        probe = mavenn.Model(L=len(self.positions), alphabet=self.alphabet, regression_type="GE",
                             gpmap_type="additive", ge_nonlinearity_type="linear",
                             ge_nonlinearity_monotonic=True, ge_nonlinearity_hidden_nodes=50,
                             ge_noise_model_type="Gaussian", theta_regularization=1e-5,
                             eta_regularization=0.001, normalize_phi=False)
        np.random.seed(seed)
        probe.set_data(self.sequences[train_indices], points[train_indices], validation_frac=0,
                       shuffle=False, verbose=False)
        only_single_mutants_found = bool(probe.x_stats["only_single_mutants"])
        del probe
        tf.keras.backend.clear_session()
        gc.collect()
        ge_nonlinearity_type = "linear" if only_single_mutants_found else "nonlinear"
        gpmap_type = "additive" if self.order == 1 or only_single_mutants_found else "custom"
        if epsilon_components is not None:
            component_rows, truth = epsilon_components
            assert set(component_rows[:, 3]) <= set(validation_indices)
            evaluation_rows = np.unique(component_rows)
            component_lookup = np.searchsorted(evaluation_rows, component_rows)
        selected, best_trial, trials = None, (float("inf"), float("inf")), []
        learning_rates = [5e-6, 5e-5, 0.005] if self.order == 2 else [0.0005, 0.001, 0.005]
        if gpmap_type == "custom":
            from epistasis.mavenn_pairwise import BilinearPairwiseGPMapLayer
            assert json.loads((ROOT / "runs/mavenn_pairwise_verification/manifest.json").read_text())["complete"]
        for learning_rate in learning_rates:
            for penalty in [1e-5, 1e-4, 1e-3, 1e-2]:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                map_options = {"gpmap_type": "additive"}
                if gpmap_type == "custom":
                    map_options = {"gpmap_type": "custom", "custom_gpmap": BilinearPairwiseGPMapLayer,
                                   "gpmap_kwargs": {"L": len(self.positions), "alphabet": self.alphabet,
                                                    "theta_regularization": penalty}}
                model = mavenn.Model(L=len(self.positions), alphabet=self.alphabet, regression_type="GE",
                                     **map_options,
                                     ge_nonlinearity_type=ge_nonlinearity_type, ge_nonlinearity_monotonic=True,
                                     ge_nonlinearity_hidden_nodes=50, ge_noise_model_type="Gaussian",
                                     theta_regularization=penalty, eta_regularization=0.001, normalize_phi=False)
                np.random.seed(seed)
                model.set_data(self.sequences[train_indices], points[train_indices], validation_frac=0,
                               shuffle=False, verbose=False)
                assert model.x_stats["only_single_mutants"] == only_single_mutants_found
                np.testing.assert_allclose(model.y_mean, np.mean(points[train_indices]), atol=1e-10)
                np.testing.assert_allclose(model.y_std, np.std(points[train_indices]), atol=1e-10)
                theta = lsmr(sparse.csc_matrix(model.x_ohe), model.y_norm.ravel(), show=False)[0]
                parameters = {"theta_0": 0.0, "theta_lc": theta}
                if gpmap_type == "custom":
                    parameters["theta_lclc"] = np.zeros((len(self.positions), 20, len(self.positions), 20), dtype=np.float32)
                model.model.x_to_phi_layer.set_params(**parameters)
                core = model.model.model
                core.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                             loss=lambda y_true, y_pred: tf.reduce_sum(y_pred), jit_compile=False)
                y = model.y_norm.astype(np.float32)
                training = np.concatenate([model.x_ohe, y], axis=1).astype(np.float32)
                best, best_epoch, weights = (float("inf"), float("inf")), -1, None
                history, rng = [], np.random.default_rng(seed)
                for epoch in range(100):
                    count = 0
                    for start in range(0, len(train_indices), self.batch_size):
                        if start == 0:
                            permutation = rng.permutation(len(train_indices))
                        rows = permutation[start:start + self.batch_size]
                        loss = core.train_on_batch(training[rows], y[rows])
                        assert np.isfinite(loss).all()
                        count += len(rows)
                    assert count == len(train_indices)
                    predicted = self.predict(model, validation_indices)
                    rho = correlation(points[validation_indices], predicted, "spearman")
                    rmse = float(np.sqrt(np.mean((points[validation_indices] - predicted) ** 2)))
                    epsilon_rmse = None
                    if epsilon_components is not None:
                        components = self.predict(model, evaluation_rows)[component_lookup]
                        epsilon = components[:, 3] - components[:, 1] - components[:, 2] + components[:, 0]
                        epsilon_rmse = float(np.sqrt(np.mean((epsilon - truth) ** 2)))
                    score = (epsilon_rmse, rmse) if epsilon_rmse is not None else (-rho if np.isfinite(rho) else float("inf"), rmse)
                    history.append({"epoch": epoch, "learning_rate": learning_rate, "theta_regularization": penalty,
                                    "training_loss": float(loss), "validation_spearman": rho,
                                    "validation_rmse": rmse, "validation_epsilon_rmse": epsilon_rmse})
                    if score < best:
                        best, best_epoch, weights = score, epoch, core.get_weights()
                    if epoch - best_epoch >= 10:
                        break
                assert weights is not None
                core.set_weights(weights)
                pd.DataFrame(history).to_csv(directory / f"history_lr{learning_rate}_reg{penalty}.csv", index=False)
                trial = {"learning_rate": learning_rate, "theta_regularization": penalty, "best_epoch": best_epoch,
                         "primary_score": best[0], "secondary_score": best[1], "epochs": len(history)}
                trials.append(trial)
                if best < best_trial:
                    best_trial, selected = best, trial
                    core.save_weights(directory / "model.weights.h5")
                    values = self.predict(model, np.arange(len(self.sequences)))
                    values = values[self.prediction_alias]
                    count = len(self.features.metadata)
                    np.savez_compressed(directory / "predictions.npz", variants=values[:count],
                                        reference=values[count], singles=values[count + 1:])
                    np.savez_compressed(directory / "normalization.npz", center=model.y_mean, scale=model.y_std)
                pd.DataFrame(trials).to_csv(directory / "trials.csv", index=False)
                print({"mavenn_trial": trial}, flush=True)
                del core, model, weights, training
                gc.collect()
        assert selected is not None
        pd.DataFrame({"sequence_position": self.positions}).to_csv(directory / "positions.csv", index=False)
        return {"selected": selected, "trials": len(trials), "feature_order": self.order,
                "gpmap_type": gpmap_type,
                "vendor_version": importlib.metadata.version("mavenn"), "tensorflow": tf.__version__,
                "version_scope": "Maintained author implementation of the published GE model, with Keras 3 support.",
                "training_loop": "Author model layers and summed negative log likelihood; external validation controls selection.",
                "pairwise_contraction": "Equivalent matrix multiplication with full float32 precision." if gpmap_type == "custom" else None,
                "pairwise_numerical_audit": "runs/mavenn_pairwise_verification/manifest.json" if gpmap_type == "custom" else None,
                "learning_rate_selection_basis": "Training-only saturation diagnostic; include the author default 0.005." if self.order == 2 else "Fixed additive-model grid.",
                "normalization_labels": "Training observations only; external validation never enters set_data.",
                "selection_metric": "epsilon_rmse" if epsilon_components is not None else "phenotype_spearman",
                "target_labels_used": len(train_indices), "validation_labels_used": len(validation_indices),
                "source_labels_used": 0, "device": self.device, "batch_size": self.batch_size,
                "genotype_positions": len(self.positions), "only_single_mutants_in_training": only_single_mutants_found,
                "ge_nonlinearity_type": ge_nonlinearity_type,
                "counterfactuals": "Reference and every component single genotype."}
