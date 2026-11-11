"""
preprocess.py
-------------
Data loading, cleaning, and type coercion for the IBM Telco churn dataset.

Public API
----------
load_raw(path)         -> pd.DataFrame
clean(df)              -> pd.DataFrame
split(df, cfg)         -> X_train, X_test, y_train, y_test
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and return an unchanged DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns from %s", len(df), df.shape[1], path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all deterministic cleaning steps:
    - Coerce TotalCharges whitespace entries (tenure=0) to 0.0
    - Drop customerID if present
    - Assert no unexpected nulls remain

    Returns a cleaned copy; the original DataFrame is unchanged.
    """
    df = df.copy()

    # TotalCharges: brand-new customers have ' ' instead of a numeric value
    before_nulls = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    coerced = df["TotalCharges"].isna().sum() - before_nulls
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if coerced:
        logger.info("TotalCharges: coerced %d whitespace entries to 0.0", coerced)

    # Drop identifier column — carries no predictive signal
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.debug("Dropped customerID column")

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls:
        logger.warning("%d null values remain after cleaning", remaining_nulls)
    else:
        logger.info("No null values after cleaning")

    logger.info("Clean shape: %d rows × %d columns", *df.shape)
    return df


def split(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/test split.

    Parameters
    ----------
    df  : cleaned DataFrame (must contain the target column)
    cfg : config dict with keys: features.target, data.test_size, data.random_state

    Returns
    -------
    X_train, X_test, y_train, y_test  (all numpy arrays)
    """
    target      = cfg["features"]["target"]
    test_size   = cfg["data"]["test_size"]
    random_state = cfg["data"]["random_state"]

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")

    X = df.drop(columns=[target]).values.astype(float)
    y = df[target].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        stratify     = y,
        random_state = random_state,
    )

    logger.info(
        "Split: train=%d (%.1f%% churn) | test=%d (%.1f%% churn)",
        len(y_train), y_train.mean() * 100,
        len(y_test),  y_test.mean()  * 100,
    )
    return X_train, X_test, y_train, y_test
