"""
utils.py
--------
Utility functions for PhysE-Inv: data preprocessing, sequence creation,
physics proxy computation, and evaluation metrics.

Reference:
    Sampath et al., "PhysE-Inv: Physics-Encoded Inverse Modeling for
    Arctic Snow Depth Estimation." Under review, 2025.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Physics Proxy
# ---------------------------------------------------------------------------

def compute_snow_depth_proxy(
    data: np.ndarray,
    A: float = 600.0,
    B: float = 300.0,
) -> np.ndarray:
    """
    Compute snow depth proxy from ERA5 variables using a simplified
    hydrostatic balance equation.

        proxy = (siconc * A) / (A - B) + (asn * rsn) / (A - B)

    Args:
        data: Array of shape (N, 3) with columns [asn, rsn, siconc].
              asn   = snow albedo
              rsn   = snow density
              siconc= sea ice concentration
        A   : Hydrostatic constant (default 600).
        B   : Hydrostatic constant (default 300).

    Returns:
        Snow depth proxy array of shape (N, 1).
    """
    siconc_term        = (data[:, 2] * A) / (A - B)
    albedo_density_term = (data[:, 0] * data[:, 1]) / (A - B)
    return (siconc_term + albedo_density_term).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Sequence Creation
# ---------------------------------------------------------------------------

def create_sequences(
    inputs: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> tuple:
    """
    Create overlapping input-output sequence pairs for sequence-to-sequence modeling.

    Each input window of length seq_len maps to a corresponding target window
    of the same length (sequence-to-sequence formulation).

    Args:
        inputs : Input array of shape (N, features).
        targets: Target array of shape (N, 1).
        seq_len: Sequence window length.

    Returns:
        X: Input sequences of shape (num_sequences, seq_len, features).
        Y: Target sequences of shape (num_sequences, seq_len, 1).
    """
    X, Y = [], []
    for i in range(len(inputs) - seq_len + 1):
        X.append(inputs[i : i + seq_len])
        Y.append(targets[i : i + seq_len])
    return np.array(X), np.array(Y)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_and_preprocess(
    filepath: str,
    feature_cols: list = ["asn", "rsn", "siconc"],
    train_ratio: float = 0.80,
    seq_len: int = 10,
    device: torch.device = None,
) -> dict:
    """
    Full preprocessing pipeline: load CSV, split, scale, compute proxy,
    create sequences, and convert to tensors.

    Args:
        filepath    : Path to ERA5 CSV file.
        feature_cols: Column names to use as input features.
        train_ratio : Fraction of data for training.
        seq_len     : Sequence window length.
        device      : Torch device.

    Returns:
        Dictionary with keys:
            X_train, y_train, X_test, y_test (tensors)
            scaler, mean_depth, std_depth
    """
    import pandas as pd

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data       = pd.read_csv(filepath)
    s_data     = data[feature_cols].values
    split      = int(len(s_data) * train_ratio)
    train_data = s_data[:split]
    test_data  = s_data[split:]

    scaler      = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled  = scaler.transform(test_data)

    train_depth = compute_snow_depth_proxy(train_scaled)
    test_depth  = compute_snow_depth_proxy(test_scaled)

    mean_depth  = train_depth.mean()
    std_depth   = train_depth.std()
    train_depth_norm = (train_depth - mean_depth) / std_depth
    test_depth_norm  = (test_depth  - mean_depth) / std_depth

    # Use only snow density (rsn, column index 1) as model input
    X_train, y_train = create_sequences(train_scaled[:, 1:2], train_depth_norm, seq_len)
    X_test,  y_test  = create_sequences(test_scaled[:,  1:2], test_depth_norm,  seq_len)

    return {
        "X_train"   : torch.tensor(X_train, dtype=torch.float32).to(device),
        "y_train"   : torch.tensor(y_train, dtype=torch.float32).to(device),
        "X_test"    : torch.tensor(X_test,  dtype=torch.float32).to(device),
        "y_test"    : torch.tensor(y_test,  dtype=torch.float32).to(device),
        "scaler"    : scaler,
        "mean_depth": mean_depth,
        "std_depth" : std_depth,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MSE and RMSE evaluation metrics.

    Args:
        y_true: Ground truth values (flattened).
        y_pred: Predicted values (flattened).

    Returns:
        Dictionary with MSE and RMSE.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE" : mse,
        "RMSE": np.sqrt(mse),
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix = f"[{label}] " if label else ""
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v:.4f}")
