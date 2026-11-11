#!/usr/bin/env python3
"""
train.py — CLI entry point for the churn prediction pipeline.

Usage
-----
    python train.py                        # uses config.yaml in current directory
    python train.py --config config.yaml   # explicit config path
    python train.py --config config.yaml --data data/telco_churn.csv

What it does
------------
1. Loads and cleans the raw dataset
2. Builds the feature engineering and encoding pipeline
3. Performs a stratified 80/20 train/test split
4. Fits a StandardScaler on the training set only
5. Runs 5-fold stratified cross-validation on all four classifier families
6. Fits every model on the full training set
7. Evaluates each model on the held-out test set
8. Saves all fitted model artefacts, the scaler, and feature names to models/
9. Generates and saves all 8 evaluation figures to figures/
10. Prints a formatted results table to stdout
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn.preprocess import load_raw, clean, split
from churn.features   import build_features, get_feature_names, fit_scaler, scale
from churn.train      import get_models, cross_validate_models, fit_all, save_artefacts
from churn.evaluate   import evaluate_all, optimise_threshold, plot_all

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train churn prediction models and save artefacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to YAML configuration file")
    p.add_argument("--data",   default=None,
                   help="Override data.raw_path in config")
    p.add_argument("--output", default=None,
                   help="Override training.output_dir in config (model save directory)")
    p.add_argument("--figures", default=None,
                   help="Override evaluation.figures_dir in config")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", path)
    return cfg


def print_results(results_df, cv_results) -> None:
    print()
    print("=" * 80)
    print("  MODEL RESULTS SUMMARY")
    print("=" * 80)
    header = f"  {'Model':<22}  {'CV AUC':>8}  {'Test AUC':>9}  {'Precision':>10}  {'Recall':>7}  {'F1':>7}  {'MCC':>7}"
    print(header)
    print("  " + "-" * 74)
    for _, row in results_df.iterrows():
        cv_std = cv_results[row["Model"]]["test_auc"].std()
        print(
            f"  {row['Model']:<22}  "
            f"{row['CV AUC']:.4f}±{cv_std:.4f}  "
            f"{row['Test AUC']:>9.4f}  "
            f"{row['Precision']:>10.4f}  "
            f"{row['Recall']:>7.4f}  "
            f"{row['F1']:>7.4f}  "
            f"{row['MCC']:>7.4f}"
        )
    print("=" * 80)
    print(f"  Best model: {results_df.iloc[0]['Model']}  "
          f"(Test AUC = {results_df.iloc[0]['Test AUC']:.4f})")
    print("=" * 80)
    print()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.data:
        cfg["data"]["raw_path"] = args.data
    if args.output:
        cfg["training"]["output_dir"] = args.output
    if args.figures:
        cfg["evaluation"]["figures_dir"] = args.figures

    # ── 1. load & clean ───────────────────────────────────────────────────────
    logger.info("Step 1/9 — Loading and cleaning data")
    df_raw   = load_raw(cfg["data"]["raw_path"])
    df_clean = clean(df_raw)

    # ── 2. feature engineering ────────────────────────────────────────────────
    logger.info("Step 2/9 — Building features")
    df_model      = build_features(df_clean, cfg)
    feature_names = get_feature_names(df_model, cfg)
    logger.info("Feature matrix: %d features", len(feature_names))

    # ── 3. train/test split ───────────────────────────────────────────────────
    logger.info("Step 3/9 — Train/test split (%.0f%%/%.0f%%)",
                (1 - cfg["data"]["test_size"]) * 100,
                cfg["data"]["test_size"] * 100)
    X_train, X_test, y_train, y_test = split(df_model, cfg)

    # ── 4. scale ──────────────────────────────────────────────────────────────
    logger.info("Step 4/9 — Fitting StandardScaler on training set")
    scaler             = fit_scaler(X_train)
    X_train_s, X_test_s = scale(scaler, X_train, X_test)

    # ── 5. instantiate models ─────────────────────────────────────────────────
    logger.info("Step 5/9 — Instantiating models")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models     = get_models(cfg, pos_weight)

    # ── 6. cross-validation ───────────────────────────────────────────────────
    logger.info("Step 6/9 — Cross-validation (%d folds)", cfg["training"]["cv_folds"])
    cv_results = cross_validate_models(models, X_train, X_train_s, y_train, cfg)

    # ── 7. fit on full training set ───────────────────────────────────────────
    logger.info("Step 7/9 — Fitting on full training set")
    fitted_models = fit_all(models, X_train, X_train_s, y_train)

    # ── 8. evaluate on test set ───────────────────────────────────────────────
    logger.info("Step 8/9 — Evaluating on held-out test set")
    results_df, test_probas, test_preds = evaluate_all(
        fitted_models, cv_results, X_test, X_test_s, y_test
    )
    print_results(results_df, cv_results)

    best_name = results_df.iloc[0]["Model"]
    best_thresh, thresh_metrics = optimise_threshold(y_test, test_probas[best_name])
    logger.info(
        "Optimal threshold for %s: %.2f  (F1=%.4f  P=%.4f  R=%.4f)",
        best_name, best_thresh,
        thresh_metrics["f1"], thresh_metrics["precision"], thresh_metrics["recall"],
    )

    # ── 9. save artefacts & figures ───────────────────────────────────────────
    logger.info("Step 9/9 — Saving artefacts and generating figures")
    output_dir  = cfg["training"]["output_dir"]
    figures_dir = cfg["evaluation"]["figures_dir"]

    save_artefacts(fitted_models, scaler, feature_names, output_dir)

    plot_all(
        df_raw        = df_raw,
        y_test        = y_test,
        test_probas   = test_probas,
        test_preds    = test_preds,
        fitted_models = fitted_models,
        feature_names = feature_names,
        cv_results    = cv_results,
        results_df    = results_df,
        best_name     = best_name,
        figures_dir   = figures_dir,
    )

    logger.info("Training complete. Models saved to '%s/', figures to '%s/'.",
                output_dir, figures_dir)


if __name__ == "__main__":
    main()
