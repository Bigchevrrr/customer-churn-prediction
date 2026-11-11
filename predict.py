#!/usr/bin/env python3
"""
predict.py — Score new customers using a saved churn model.

Usage
-----
    # Score a CSV file and print results to stdout
    python predict.py --input data/new_customers.csv

    # Score and save predictions to a CSV
    python predict.py --input data/new_customers.csv --output predictions.csv

    # Use a specific model (default: best model by Test AUC = Gradient Boosting)
    python predict.py --input data/new_customers.csv --model gradient_boosting

    # Adjust decision threshold (default: 0.5)
    python predict.py --input data/new_customers.csv --threshold 0.42

Input format
------------
The input CSV must have the same 21 columns as the original Telco dataset
(customerID is optional and will be preserved in output if present).
Run train.py first to generate the model artefacts in models/.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy  as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn.preprocess import load_raw, clean
from churn.features   import build_features, get_feature_names, scale
from churn.train      import load_artefacts

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("predict")

MODEL_SLUGS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "xgboost":             "XGBoost",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score new customers with a saved churn model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",     required=True,
                   help="Path to CSV with new customer records")
    p.add_argument("--output",    default=None,
                   help="Path to write predictions CSV (optional; prints to stdout if omitted)")
    p.add_argument("--models_dir", default="models",
                   help="Directory containing saved model artefacts")
    p.add_argument("--config",    default="config.yaml",
                   help="Path to YAML configuration file")
    p.add_argument("--model",     default="gradient_boosting",
                   choices=list(MODEL_SLUGS.keys()),
                   help="Which saved model to use for scoring")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for churn flag (0–1)")
    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_prediction_features(
    df_input:      pd.DataFrame,
    cfg:           dict,
    feature_names: list[str],
) -> np.ndarray:
    """
    Apply the same feature engineering pipeline used during training,
    then align columns to the exact feature set the model expects.

    Missing columns are filled with 0; extra columns are dropped.
    This handles cases where new data has fewer OHE categories than training.
    """
    # temporarily inject a dummy target so build_features doesn't error
    df_input = df_input.copy()
    has_target = cfg["features"]["target"] in df_input.columns
    if not has_target:
        df_input[cfg["features"]["target"]] = "No"

    df_model  = build_features(clean(df_input), cfg)
    feat_cols = get_feature_names(df_model, cfg)
    X_raw     = df_model[feat_cols]

    # align to training feature set
    X_aligned = X_raw.reindex(columns=feature_names, fill_value=0)
    return X_aligned.values.astype(float)


def main() -> None:
    args = parse_args()

    # ── load artefacts ────────────────────────────────────────────────────────
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logger.error(
            "models/ directory not found at '%s'. "
            "Run train.py first to generate model artefacts.", models_dir
        )
        sys.exit(1)

    logger.info("Loading model artefacts from %s/", models_dir)
    fitted_models, scaler, feature_names = load_artefacts(models_dir)

    model_name = MODEL_SLUGS[args.model]
    if model_name not in fitted_models:
        logger.error("Model '%s' not found in artefacts. Available: %s",
                     model_name, list(fitted_models.keys()))
        sys.exit(1)

    model = fitted_models[model_name]
    logger.info("Using model: %s  |  threshold: %.2f", model_name, args.threshold)

    # ── load & process input ──────────────────────────────────────────────────
    cfg      = load_config(args.config)
    df_input = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(df_input), args.input)

    # preserve customerID if present
    customer_ids = df_input["customerID"].values if "customerID" in df_input.columns else None

    # build features and scale
    X = build_prediction_features(df_input, cfg, feature_names)
    if model_name == "Logistic Regression":
        _, X = scale(scaler, X, X)   # use transform only
        X_score = X
    else:
        X_score = X

    # ── score ─────────────────────────────────────────────────────────────────
    probas = model.predict_proba(X_score)[:, 1]
    flags  = (probas >= args.threshold).astype(int)
    risk   = pd.cut(
        probas,
        bins   = [0, 0.3, 0.5, 0.7, 1.0],
        labels = ["Low", "Medium", "High", "Critical"],
        right  = True,
    )

    # ── build output DataFrame ────────────────────────────────────────────────
    out = pd.DataFrame({
        "churn_probability": probas.round(4),
        "churn_flag":        flags,
        "risk_tier":         risk,
    })
    if customer_ids is not None:
        out.insert(0, "customerID", customer_ids)

    # summary
    n = len(out)
    n_flagged = flags.sum()
    logger.info(
        "Scored %d customers: %d flagged for churn (%.1f%%) at threshold %.2f",
        n, n_flagged, n_flagged / n * 100, args.threshold,
    )
    for tier in ["Critical", "High", "Medium", "Low"]:
        cnt = (risk == tier).sum()
        logger.info("  %s risk: %d  (%.1f%%)", tier, cnt, cnt/n*100)

    # ── output ────────────────────────────────────────────────────────────────
    if args.output:
        out.to_csv(args.output, index=False)
        logger.info("Predictions saved to %s", args.output)
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
