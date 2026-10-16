import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
INPUT_CSV = Path("keypoints_pairs.csv")
REPORT_CSV = Path("map_fit_report.csv")
PREDICTIONS_CSV = Path("keypoint_predictions.csv")
MODEL_FILE = Path("transport_maps.pkl")
RBF_REGULARIZATION = 1e-8
GP_NOISE = 1e-8
# --------------------------------------------------
# RIGID TRANSFORMATION: KABSCH ALGORITHM
# --------------------------------------------------
def fit_rigid(X, Y):
    """Fit Y ~= (X - source_center) @ R + target_center."""
    source_center = X.mean(axis=0)
    target_center = Y.mean(axis=0)
    Xc = X - source_center
    Yc = Y - target_center
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(U @ Vt)
    R = U @ correction @ Vt
    return {
        "source_center": source_center,
        "target_center": target_center,
        "R": R,
    }
def predict_rigid(model, X):
    return (
        (X - model["source_center"]) @ model["R"]
        + model["target_center"]
    )
# --------------------------------------------------
# RBF DISPLACEMENT MAP
# --------------------------------------------------
def gaussian_kernel(A, B, length_scale):
    distances = cdist(A, B)
    return np.exp(-0.5 * (distances / length_scale) ** 2)
def fit_rbf(X, Y):
    """Fit a smooth Gaussian RBF model to displacement Y - X."""
    displacement = Y - X
    distances = cdist(X, X)
    nonzero = distances[distances > 1e-12]
    length_scale = np.median(nonzero)
    K = gaussian_kernel(X, X, length_scale)
    weights = np.linalg.solve(
        K + RBF_REGULARIZATION * np.eye(len(X)),
        displacement,
    )
    return {
        "source_points": X.copy(),
        "length_scale": length_scale,
        "weights": weights,
    }
def predict_rbf(model, X):
    K_query = gaussian_kernel(
        X,
        model["source_points"],
        model["length_scale"],
    )
    return X + K_query @ model["weights"]
# --------------------------------------------------
# GAUSSIAN PROCESS DISPLACEMENT MAP
# --------------------------------------------------
def fit_gp(X, Y):
    displacement = Y - X
    distances = cdist(X, X)
    nonzero = distances[distances > 1e-12]
    length_scale = np.median(nonzero)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=length_scale, length_scale_bounds="fixed")
    )
    models = []
    for axis in range(3):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=GP_NOISE,
            optimizer=None,
            normalize_y=False,
        )
        gp.fit(X, displacement[:, axis])
        models.append(gp)
    return {
        "models": models,
        "length_scale": length_scale,
    }
def predict_gp(model, X):
    displacement = np.column_stack([
        gp.predict(X)
        for gp in model["models"]
    ])
    return X + displacement
# --------------------------------------------------
# ERROR METRICS
# --------------------------------------------------
def calculate_errors(predicted, actual):
    errors = np.linalg.norm(predicted - actual, axis=1)
    return {
        "train_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "train_mean_error_m": float(np.mean(errors)),
        "train_max_error_m": float(np.max(errors)),
    }
# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    df = pd.read_csv(INPUT_CSV)
    required = {
        "object", "point_id",
        "src_x", "src_y", "src_z",
        "tgt_x", "tgt_y", "tgt_z",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if df.duplicated(["object", "point_id"]).any():
        raise ValueError("Duplicate object/point_id pairs found.")
    if df[list(required - {"object", "point_id"})].isna().any().any():
        raise ValueError("Missing numeric coordinates in CSV.")
    report_rows = []
    prediction_frames = []
    saved_models = {}
    for obj, group in df.groupby("object"):
        group = group.reset_index(drop=True)
        X = group[["src_x", "src_y", "src_z"]].to_numpy(float)
        Y = group[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(float)
        if len(X) < 4:
            raise ValueError(
                f"{obj}: at least four keypoints are recommended."
            )
        if np.linalg.matrix_rank(X - X.mean(axis=0)) < 2:
            raise ValueError(
                f"{obj}: source keypoints are collinear or coincident; "
                "a 3D rigid transformation cannot be reliably fitted."
            )
        models = {
            "rigid": fit_rigid(X, Y),
            "rbf": fit_rbf(X, Y),
            "gp": fit_gp(X, Y),
        }
        predictors = {
            "rigid": predict_rigid(models["rigid"], X),
            "rbf": predict_rbf(models["rbf"], X),
            "gp": predict_gp(models["gp"], X),
        }
        saved_models[obj] = models
        pred = group[["object", "point_id"]].copy()
        for axis, j in zip("xyz", range(3)):
            pred[f"src_{axis}"] = X[:, j]
            pred[f"tgt_{axis}"] = Y[:, j]
        for method, predicted in predictors.items():
            metrics = calculate_errors(predicted, Y)
            report_rows.append({
                "object": obj,
                "method": method,
                "num_keypoints": len(X),
                **metrics,
            })
            for j, axis in enumerate("xyz"):
                pred[f"{method}_{axis}"] = predicted[:, j]
            pred[f"{method}_error_m"] = np.linalg.norm(
                predicted - Y, axis=1
            )
        prediction_frames.append(pred)
    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        PREDICTIONS_CSV, index=False
    )
    with MODEL_FILE.open("wb") as f:
        pickle.dump(saved_models, f)
    print("\n=== Training error report ===")
    print(pd.DataFrame(report_rows).to_string(index=False))
    print(f"\nSaved report:      {REPORT_CSV.resolve()}")
    print(f"Saved predictions: {PREDICTIONS_CSV.resolve()}")
    print(f"Saved models:      {MODEL_FILE.resolve()}")
    print(
        "\nImportant: these are training errors at the same keypoints "
        "used to fit the maps, not independent generalization errors."
    )
if __name__ == "__main__":
    main()
