"""
features.py
-----------
Feature engineering and encoding pipeline for the churn dataset.

Steps applied in order:
1. Binary encode Yes/No columns and gender
2. One-hot encode nominal multi-class columns (drop_first=True)
3. Engineer three domain-specific features:
       avg_monthly_spend  = TotalCharges / (tenure + 1)
       is_new_customer    = 1 if tenure < 12 else 0
       services_count     = sum of 6 add-on service flags
4. Fit StandardScaler on training set only (no data leakage)

Public API
----------
build_features(df_clean, cfg)         -> df_model (encoded DataFrame)
get_feature_names(df_model, cfg)      -> list[str]
fit_scaler(X_train)                   -> StandardScaler
scale(scaler, X_train, X_test)        -> X_train_s, X_test_s
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy  as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _apply_binary_encoding(df: pd.DataFrame, binary_col_maps: dict) -> pd.DataFrame:
    """Map binary categorical columns to 0/1 integers."""
    for col, mapping in binary_col_maps.items():
        if col in df.columns:
            # cast to object so .map() returns np.nan (not pd.NA) on misses
            df[col] = df[col].astype(object).map(mapping)
            n_na = df[col].isna().sum()
            if n_na:
                bad = df.loc[df[col].isna()].index.tolist()[:5]
                raise ValueError(
                    f"Binary encoding for '{col}' produced {n_na} NaNs "
                    f"(row indices: {bad}). Check mapping keys."
                )
    return df


def _apply_ohe(df: pd.DataFrame, ohe_cols: list[str]) -> pd.DataFrame:
    """One-hot encode nominal columns with drop_first=True to avoid multicollinearity."""
    present = [c for c in ohe_cols if c in df.columns]
    missing = [c for c in ohe_cols if c not in df.columns]
    if missing:
        logger.warning("OHE columns not found and skipped: %s", missing)
    df = pd.get_dummies(df, columns=present, drop_first=True)
    return df


def _engineer_features(df: pd.DataFrame, raw_df: pd.DataFrame, service_cols: list[str]) -> pd.DataFrame:
    """
    Add three engineered features.

    avg_monthly_spend : normalises lifetime spend by tenure — removes the
                        confound between account age and total revenue.
    is_new_customer   : binary flag for first-year customers, the highest-
                        risk acquisition cohort.
    services_count    : count of add-on services subscribed — each additional
                        service raises switching cost and lowers churn risk.
    """
    df["avg_monthly_spend"] = (df["TotalCharges"] / (df["tenure"] + 1)).round(4)
    df["is_new_customer"]   = (df["tenure"] < 12).astype(int)

    # service flags must be read from raw (pre-encoding) DataFrame
    svc_flags = sum((raw_df[c] == "Yes").astype(int) for c in service_cols if c in raw_df.columns)
    df["services_count"] = svc_flags.values if hasattr(svc_flags, "values") else svc_flags

    logger.info(
        "Engineered features: avg_monthly_spend (mean=%.3f), "
        "is_new_customer (%%new=%.1f%%), services_count (mean=%.2f)",
        df["avg_monthly_spend"].mean(),
        df["is_new_customer"].mean() * 100,
        df["services_count"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_features(df_clean: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply the full feature engineering and encoding pipeline.

    Parameters
    ----------
    df_clean : output of preprocess.clean() — must NOT contain customerID
    cfg      : full config dict (uses features.* keys)

    Returns
    -------
    df_model : fully encoded DataFrame ready for train/test split
               (target column is encoded to 0/1 int)
    """
    df       = df_clean.copy()
    raw_copy = df_clean.copy()   # keep original category labels for service counting

    feat_cfg     = cfg["features"]
    target       = feat_cfg["target"]
    binary_maps  = feat_cfg["binary_cols"]
    ohe_cols     = feat_cfg["ohe_cols"]
    service_cols = feat_cfg["service_cols"]

    # encode target
    df[target] = df[target].map({"Yes": 1, "No": 0})
    if df[target].isna().any():
        raise ValueError("Target column contains values other than 'Yes'/'No'")

    # binary encoding
    df = _apply_binary_encoding(df, binary_maps)

    # OHE
    df = _apply_ohe(df, ohe_cols)

    # domain feature engineering
    df = _engineer_features(df, raw_copy, service_cols)

    n_features = df.shape[1] - 1   # exclude target
    logger.info("Feature matrix: %d rows × %d features", len(df), n_features)
    return df


def get_feature_names(df_model: pd.DataFrame, cfg: dict) -> list[str]:
    """Return ordered list of feature column names (excluding target)."""
    target = cfg["features"]["target"]
    return [c for c in df_model.columns if c != target]


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training data only. Never call on test data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info(
        "Scaler fitted on %d training samples. "
        "Feature means range: [%.4f, %.4f]",
        len(X_train),
        scaler.mean_.min(),
        scaler.mean_.max(),
    )
    return scaler


def scale(
    scaler: StandardScaler,
    X_train: np.ndarray,
    X_test:  Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply a fitted scaler to train and optionally test arrays.

    Parameters
    ----------
    scaler  : already-fitted StandardScaler
    X_train : training feature matrix
    X_test  : test feature matrix (optional)

    Returns
    -------
    (X_train_scaled, X_test_scaled)  — X_test_scaled is None if X_test is None
    """
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test) if X_test is not None else None
    return X_train_s, X_test_s
