"""
train.py
--------
Model definitions, cross-validation, and fitting.

Public API
----------
get_models(cfg, pos_weight)           -> dict[str, estimator]
cross_validate_models(models, ...)    -> dict[str, dict]
fit_all(models, X_train, y_train)     -> dict[str, fitted_estimator]
save_artefacts(models, scaler, feature_names, output_dir)
load_artefacts(output_dir)            -> models, scaler, feature_names
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from sklearn.ensemble        import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing   import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# model factory
# ---------------------------------------------------------------------------

def get_models(cfg: dict, pos_weight: float) -> dict[str, Any]:
    """
    Instantiate all four classifier families from config.

    Parameters
    ----------
    cfg        : full config dict
    pos_weight : ratio of negative to positive class samples
                 (used by XGBoost's scale_pos_weight)

    Returns
    -------
    Ordered dict: name -> unfitted sklearn-compatible estimator
    """
    mc = cfg["models"]

    models = {
        "Logistic Regression": LogisticRegression(
            C            = mc["logistic_regression"]["C"],
            max_iter     = mc["logistic_regression"]["max_iter"],
            solver       = mc["logistic_regression"]["solver"],
            class_weight = mc["logistic_regression"]["class_weight"],
            random_state = mc["logistic_regression"]["random_state"],
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators   = mc["random_forest"]["n_estimators"],
            max_depth      = mc["random_forest"]["max_depth"],
            min_samples_leaf = mc["random_forest"]["min_samples_leaf"],
            max_features   = mc["random_forest"]["max_features"],
            class_weight   = mc["random_forest"]["class_weight"],
            n_jobs         = mc["random_forest"]["n_jobs"],
            random_state   = mc["random_forest"]["random_state"],
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators   = mc["gradient_boosting"]["n_estimators"],
            learning_rate  = mc["gradient_boosting"]["learning_rate"],
            max_depth      = mc["gradient_boosting"]["max_depth"],
            subsample      = mc["gradient_boosting"]["subsample"],
            min_samples_leaf = mc["gradient_boosting"]["min_samples_leaf"],
            max_features   = mc["gradient_boosting"]["max_features"],
            random_state   = mc["gradient_boosting"]["random_state"],
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators      = mc["xgboost"]["n_estimators"],
            learning_rate     = mc["xgboost"]["learning_rate"],
            max_depth         = mc["xgboost"]["max_depth"],
            subsample         = mc["xgboost"]["subsample"],
            colsample_bytree  = mc["xgboost"]["colsample_bytree"],
            reg_alpha         = mc["xgboost"]["reg_alpha"],
            reg_lambda        = mc["xgboost"]["reg_lambda"],
            scale_pos_weight  = pos_weight,
            eval_metric       = mc["xgboost"]["eval_metric"],
            random_state      = mc["xgboost"]["random_state"],
            verbosity         = mc["xgboost"]["verbosity"],
            n_jobs            = mc["xgboost"]["n_jobs"],
            use_label_encoder = False,
        ),
    }

    logger.info("Instantiated %d models: %s", len(models), list(models.keys()))
    return models


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------

def cross_validate_models(
    models:   dict[str, Any],
    X_train:  np.ndarray,
    X_train_s: np.ndarray,
    y_train:  np.ndarray,
    cfg:      dict,
) -> dict[str, dict]:
    """
    Run stratified k-fold cross-validation on all models.

    Logistic Regression receives the scaled feature matrix; tree-based
    models receive the unscaled matrix (they are scale-invariant).

    Parameters
    ----------
    models    : dict of unfitted estimators from get_models()
    X_train   : unscaled training features
    X_train_s : StandardScaler-transformed training features
    y_train   : training labels
    cfg       : full config dict

    Returns
    -------
    cv_results : dict mapping model name -> cross_validate output dict
    """
    n_splits     = cfg["training"]["cv_folds"]
    cv_strategy  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring      = {"auc": "roc_auc", "avg_precision": "average_precision", "f1": "f1"}
    cv_results   = {}

    logger.info("Starting %d-fold stratified cross-validation ...", n_splits)

    for name, model in models.items():
        X_cv = X_train_s if name == "Logistic Regression" else X_train
        t0   = time()
        scores = cross_validate(model, X_cv, y_train, cv=cv_strategy,
                                scoring=scoring, n_jobs=-1)
        elapsed = time() - t0
        cv_results[name] = scores

        logger.info(
            "%s  |  AUC=%.4f±%.4f  AvgP=%.4f  F1=%.4f  [%.1fs]",
            name,
            scores["test_auc"].mean(), scores["test_auc"].std(),
            scores["test_avg_precision"].mean(),
            scores["test_f1"].mean(),
            elapsed,
        )

    best = max(cv_results, key=lambda n: cv_results[n]["test_auc"].mean())
    logger.info("Best CV model: %s (AUC=%.4f)", best,
                cv_results[best]["test_auc"].mean())
    return cv_results


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def fit_all(
    models:   dict[str, Any],
    X_train:  np.ndarray,
    X_train_s: np.ndarray,
    y_train:  np.ndarray,
) -> dict[str, Any]:
    """
    Fit every model on the full training set.

    Returns a new dict of fitted estimators (originals are unchanged).
    """
    fitted = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "Logistic Regression" else X_train
        t0   = time()
        model.fit(X_tr, y_train)
        fitted[name] = model
        logger.info("Fitted %s in %.1fs", name, time() - t0)
    return fitted


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_artefacts(
    fitted_models: dict[str, Any],
    scaler:        StandardScaler,
    feature_names: list[str],
    output_dir:    str | Path,
) -> None:
    """
    Serialise all fitted models, the scaler, and the feature name list
    to the output directory using pickle.

    Layout
    ------
    output_dir/
        logistic_regression.pkl
        random_forest.pkl
        gradient_boosting.pkl
        xgboost.pkl
        scaler.pkl
        feature_names.pkl
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest":       "random_forest",
        "Gradient Boosting":   "gradient_boosting",
        "XGBoost":             "xgboost",
    }

    for name, model in fitted_models.items():
        slug = slug_map.get(name, name.lower().replace(" ", "_"))
        path = output_dir / f"{slug}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved %s -> %s", name, path)

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler -> %s/scaler.pkl", output_dir)

    with open(output_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    logger.info("Saved feature_names (%d features) -> %s/feature_names.pkl",
                len(feature_names), output_dir)


def load_artefacts(output_dir: str | Path) -> tuple[dict, StandardScaler, list[str]]:
    """
    Load all fitted artefacts previously saved with save_artefacts().

    Returns
    -------
    (fitted_models, scaler, feature_names)
    """
    output_dir = Path(output_dir)
    slug_map   = {
        "logistic_regression": "Logistic Regression",
        "random_forest":       "Random Forest",
        "gradient_boosting":   "Gradient Boosting",
        "xgboost":             "XGBoost",
    }

    fitted_models = {}
    for slug, name in slug_map.items():
        path = output_dir / f"{slug}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                fitted_models[name] = pickle.load(f)
            logger.info("Loaded %s from %s", name, path)

    with open(output_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(output_dir / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    logger.info("Loaded %d models, scaler, %d feature names",
                len(fitted_models), len(feature_names))
    return fitted_models, scaler, feature_names
