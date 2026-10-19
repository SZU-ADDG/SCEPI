import gc
import json
from pathlib import Path
import subprocess
import sys

import mavenn
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.higher import HigherFeatures
from epistasis.mavenn import MavennRegression
from epistasis.mavenn_pairwise import BilinearPairwiseGPMapLayer

features = HigherFeatures(ROOT / "data/features/higher/functional_doubles/func_0000")
regression = MavennRegression(features, "functional_doubles", 2, "cuda")
ledger = pd.read_parquet(ROOT / "runs/mavenn_smoke/functional_doubles/order2/func_0000/seed11/random/C128/label_permissions.parquet")
lookup = pd.Series(np.arange(len(features.metadata)), index=features.metadata.variant_id)
training_rows = lookup.loc[ledger.loc[ledger.role.eq("train"), "variant_id"]].to_numpy()
points = features.metadata.point.to_numpy()
common = {"L": len(regression.positions), "alphabet": regression.alphabet, "regression_type": "GE",
          "ge_nonlinearity_type": "nonlinear", "ge_nonlinearity_monotonic": True,
          "ge_nonlinearity_hidden_nodes": 50, "ge_noise_model_type": "Gaussian",
          "theta_regularization": 1e-4, "eta_regularization": 0.001, "normalize_phi": False}
custom = {"gpmap_type": "custom", "custom_gpmap": BilinearPairwiseGPMapLayer,
          "gpmap_kwargs": {"L": common["L"], "alphabet": common["alphabet"], "theta_regularization": 1e-4}}
tf.keras.utils.set_random_seed(11)
original = mavenn.Model(**common, gpmap_type="pairwise")
original.set_data(regression.sequences[training_rows], points[training_rows], validation_frac=0, shuffle=False, verbose=False)
theta = lsmr(sparse.csc_matrix(original.x_ohe), original.y_norm.ravel(), show=False)[0]
original.model.x_to_phi_layer.set_params(theta_0=0.0, theta_lc=theta,
    theta_lclc=np.zeros((common["L"], 20, common["L"], 20), dtype=np.float32))
accelerated = mavenn.Model(**common, **custom)
accelerated.set_data(regression.sequences[training_rows], points[training_rows], validation_frac=0, shuffle=False, verbose=False)
np.testing.assert_array_equal(original.x_ohe, regression.ohe(training_rows))
a, b = original.model.model, accelerated.model.model
b.set_weights(a.get_weights())
initial = a.get_weights()
inputs = np.concatenate([original.x_ohe, original.y_norm.astype(np.float32)], axis=1).astype(np.float32)
optimizers = [tf.keras.optimizers.Adam(1e-5), tf.keras.optimizers.Adam(1e-5)]
checks = []
for step in range(3):
    current = tf.convert_to_tensor(inputs[step * 4:(step + 1) * 4])
    outputs, gradients, losses = [], [], []
    for core in [a, b]:
        with tf.GradientTape() as tape:
            output = core(current, training=True)
            loss = tf.reduce_sum(output) + tf.add_n(core.losses)
        gradient = tape.gradient(loss, core.trainable_variables)
        assert all(value is not None and np.isfinite(value.numpy()).all() for value in gradient)
        outputs.append(output.numpy())
        gradients.append(gradient)
        losses.append(float(loss))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)
    maximum = 0.0
    for first, second in zip(*gradients):
        np.testing.assert_allclose(first.numpy(), second.numpy(), rtol=1e-5, atol=1e-5)
        maximum = max(maximum, float(np.max(np.abs(first.numpy() - second.numpy()))))
    for core, optimizer, gradient in zip([a, b], optimizers, gradients):
        optimizer.apply_gradients(zip(gradient, core.trainable_variables))
    checks.append({"step": step, "maximum_forward_difference": float(np.max(np.abs(outputs[0] - outputs[1]))),
                   "maximum_gradient_difference": maximum, "trainable_tensors": len(gradients[0]), "losses": losses})
del original, a, optimizers, gradients
gc.collect()
diagnostics = []
for learning_rate in [0.000005, 0.00005, 0.0005, 0.005]:
    b.set_weights(initial)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    for step in range(3):
        rows = np.arange(step * 50, min((step + 1) * 50, len(inputs)))
        current = tf.convert_to_tensor(inputs[rows])
        before = accelerated.model.x_to_phi_layer(current[:, :-1]).numpy()
        with tf.GradientTape() as tape:
            output = b(current, training=True)
            loss = tf.reduce_sum(output) + tf.add_n(b.losses)
        gradient = tape.gradient(loss, b.trainable_variables)
        optimizer.apply_gradients(zip(gradient, b.trainable_variables))
        after = accelerated.model.x_to_phi_layer(current[:, :-1]).numpy()
        prediction = accelerated.model.phi_to_yhat_layer(tf.convert_to_tensor(after)).numpy()
        diagnostics.append({"learning_rate": learning_rate, "step": step,
                            "training_latent_min_before": float(before.min()), "training_latent_max_before": float(before.max()),
                            "training_latent_min_after": float(after.min()), "training_latent_max_after": float(after.max()),
                            "training_prediction_std_after": float(prediction.std()), "training_loss": float(loss)})
output = ROOT / "runs/mavenn_pairwise_verification"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(checks).to_json(output / "gradient_checks.json", orient="records", indent=2)
pd.DataFrame(diagnostics).to_csv(output / "training_conditioning.csv", index=False)
manifest = {"complete": True, "source_commit": subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "actual_training_variants": len(training_rows), "genotype_positions": common["L"],
    "gradient_steps": len(checks), "maximum_gradient_difference": max(row["maximum_gradient_difference"] for row in checks),
    "maximum_forward_difference": max(row["maximum_forward_difference"] for row in checks),
    "onehot_encoding_matches_author": True,
    "labels_used": "Only the permitted C128 training labels of func_0000, seed 11, random split.",
    "evaluation_labels_used": 0,
    "equivalence": "Original MAVE-NN pairwise parameters, mask, regularizers, GE and noise layers; the same pairwise contraction uses matrix multiplication.",
    "training_diagnostics": str((output / "training_conditioning.csv").relative_to(ROOT))}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
