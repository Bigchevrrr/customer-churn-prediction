"""
evaluate.py
-----------
Metrics computation, threshold optimisation, and chart generation.

Public API
----------
compute_metrics(y_true, y_pred, y_proba)   -> dict
evaluate_all(fitted_models, ...)           -> pd.DataFrame, dict[probas]
optimise_threshold(y_true, y_proba)        -> best_threshold, metrics_at_threshold
plot_all(y_test, test_probas, fitted_models,
         feature_names, cv_results, figures_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing  import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import numpy  as np
import pandas as pd
from sklearn.calibration  import calibration_curve
from sklearn.inspection   import permutation_importance
from sklearn.metrics      import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

logger = logging.getLogger(__name__)

# ── design system ──────────────────────────────────────────────────────────────
BG    = "#0a0f1e"
CARD  = "#0e1628"
FG    = "#cdd6f4"
MUTED = "#7a8bba"
BLUE  = "#4f8ef7"
RED   = "#c0392b"
GREEN = "#2ecc71"
GOLD  = "#f39c12"
DIV   = "#1a2540"
PURP  = "#9b59b6"
TEAL  = "#1abc9c"
PAL   = [BLUE, RED, GREEN, GOLD, PURP, TEAL]

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":  CARD,
    "axes.edgecolor":   DIV,  "axes.labelcolor": MUTED,
    "xtick.color":      MUTED,"ytick.color":     MUTED,
    "text.color":       FG,   "grid.color":      DIV,
    "grid.linewidth":   0.5,  "figure.dpi":      110,
    "font.size":        10,   "axes.titlepad":   10,
    "legend.framealpha":0.85, "legend.edgecolor":DIV,
    "lines.linewidth":  1.8,
})


def _save(name: str, figures_dir: Path, dpi: int = 130) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.info("Saved figure: %s", path)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Return a dict of all evaluation metrics for one model."""
    return {
        "Test AUC":    round(roc_auc_score(y_true, y_proba), 4),
        "Avg Prec":    round(average_precision_score(y_true, y_proba), 4),
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Precision":   round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":      round(recall_score(y_true, y_pred), 4),
        "F1":          round(f1_score(y_true, y_pred), 4),
        "MCC":         round(matthews_corrcoef(y_true, y_pred), 4),
        "Brier Score": round(brier_score_loss(y_true, y_proba), 4),
    }


def evaluate_all(
    fitted_models: dict[str, Any],
    cv_results:    dict[str, dict],
    X_test:        np.ndarray,
    X_test_s:      np.ndarray,
    y_test:        np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Evaluate every fitted model on the held-out test set.

    Returns
    -------
    results_df   : DataFrame sorted by Test AUC descending
    test_probas  : dict[name -> probability array]
    test_preds   : dict[name -> binary prediction array]
    """
    rows        = []
    test_probas = {}
    test_preds  = {}

    for name, model in fitted_models.items():
        X_te   = X_test_s if name == "Logistic Regression" else X_test
        proba  = model.predict_proba(X_te)[:, 1]
        preds  = model.predict(X_te)
        test_probas[name] = proba
        test_preds[name]  = preds

        metrics = compute_metrics(y_test, preds, proba)
        metrics["Model"]  = name
        metrics["CV AUC"] = round(cv_results[name]["test_auc"].mean(), 4)
        rows.append(metrics)

        logger.info(
            "%s | AUC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f",
            name, metrics["Test AUC"], metrics["F1"],
            metrics["Precision"], metrics["Recall"],
        )

    col_order  = ["Model", "CV AUC", "Test AUC", "Avg Prec",
                  "Accuracy", "Precision", "Recall", "F1", "MCC", "Brier Score"]
    results_df = (pd.DataFrame(rows)[col_order]
                    .sort_values("Test AUC", ascending=False)
                    .reset_index(drop=True))
    return results_df, test_probas, test_preds


# ---------------------------------------------------------------------------
# threshold optimisation
# ---------------------------------------------------------------------------

def optimise_threshold(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    step:    float = 0.01,
) -> tuple[float, dict]:
    """
    Sweep decision thresholds and return the one that maximises F1.

    Returns
    -------
    best_threshold : float
    metrics        : dict with precision, recall, f1 at that threshold
    """
    thresholds = np.arange(0.05, 0.95, step)
    best_f1    = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = thresh

    opt_preds = (y_proba >= best_thresh).astype(int)
    metrics   = {
        "threshold": round(float(best_thresh), 2),
        "precision": round(precision_score(y_true, opt_preds, zero_division=0), 4),
        "recall":    round(recall_score(y_true, opt_preds), 4),
        "f1":        round(f1_score(y_true, opt_preds), 4),
    }
    logger.info(
        "Optimal threshold: %.2f | P=%.4f  R=%.4f  F1=%.4f",
        best_thresh, metrics["precision"], metrics["recall"], metrics["f1"],
    )
    return best_thresh, metrics


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------

def plot_roc_pr(
    y_test:      np.ndarray,
    test_probas: dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("ROC and Precision-Recall Curves — All Models",
                 color=FG, fontsize=13, fontweight="bold", y=1.01)

    axes[0].set_facecolor(CARD)
    for i, (name, proba) in enumerate(test_probas.items()):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_v = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, color=PAL[i], linewidth=2.0,
                     label=f"{name}  (AUC={auc_v:.4f})")
    axes[0].plot([0,1],[0,1], "--", color=MUTED, linewidth=1.0,
                 alpha=0.5, label="Random (AUC=0.5000)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves", color=FG, fontsize=12, fontweight="bold")
    axes[0].legend(facecolor=CARD, edgecolor=DIV, labelcolor=FG, fontsize=8.5, loc="lower right")
    for sp in axes[0].spines.values(): sp.set_edgecolor(DIV)
    axes[0].grid(color=DIV, linewidth=0.5)

    axes[1].set_facecolor(CARD)
    baseline = y_test.mean()
    for i, (name, proba) in enumerate(test_probas.items()):
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        axes[1].plot(rec_arr, prec_arr, color=PAL[i], linewidth=2.0,
                     label=f"{name}  (AP={ap:.4f})")
    axes[1].axhline(baseline, color=MUTED, linestyle="--", linewidth=1.0,
                    alpha=0.6, label=f"Baseline ({baseline:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves", color=FG, fontsize=12, fontweight="bold")
    axes[1].legend(facecolor=CARD, edgecolor=DIV, labelcolor=FG, fontsize=8.5)
    for sp in axes[1].spines.values(): sp.set_edgecolor(DIV)
    axes[1].grid(color=DIV, linewidth=0.5)

    plt.tight_layout()
    _save("05_roc_curves.png", figures_dir)


def plot_confusion_and_metrics(
    y_test:     np.ndarray,
    best_preds: np.ndarray,
    best_proba: np.ndarray,
    best_name:  str,
    figures_dir: Path,
) -> None:
    cm = confusion_matrix(y_test, best_preds)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Confusion Matrix and Metrics — {best_name}",
                 color=FG, fontsize=12, fontweight="bold", y=1.01)

    im = axes[0].imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=axes[0])
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            lbl = [["TN","FP"],["FN","TP"]][i][j]
            axes[0].text(j, i, f"{lbl}\n{v:,}", ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color=FG if v < cm.max()*0.6 else BG)
    axes[0].set_xticks([0,1]); axes[0].set_xticklabels(["Pred: No","Pred: Yes"])
    axes[0].set_yticks([0,1]); axes[0].set_yticklabels(["Actual: No","Actual: Yes"])
    axes[0].set_title("Confusion Matrix", color=FG, fontsize=11, fontweight="bold")
    axes[0].set_facecolor(CARD)
    for sp in axes[0].spines.values(): sp.set_edgecolor(DIV)

    metric_names = ["AUC","Avg Prec","Precision","Recall","F1","Accuracy"]
    metric_vals  = [
        roc_auc_score(y_test, best_proba),
        average_precision_score(y_test, best_proba),
        precision_score(y_test, best_preds, zero_division=0),
        recall_score(y_test, best_preds),
        f1_score(y_test, best_preds),
        accuracy_score(y_test, best_preds),
    ]
    bars = axes[1].barh(metric_names[::-1], metric_vals[::-1],
                        color=PAL[:len(metric_names)][::-1], edgecolor="none", height=0.55)
    for bar, v in zip(bars, metric_vals[::-1]):
        axes[1].text(v + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{v:.4f}", va="center", fontsize=10, color=FG, fontweight="bold")
    axes[1].set_xlim([0, 1.08])
    axes[1].set_facecolor(CARD)
    axes[1].set_title("Classification Metrics", color=FG, fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Score")
    for sp in axes[1].spines.values(): sp.set_edgecolor(DIV)
    axes[1].grid(axis="x", color=DIV, linewidth=0.5)

    plt.tight_layout()
    _save("06_confusion_matrix.png", figures_dir)


def plot_feature_importance(
    fitted_models: dict[str, Any],
    feature_names: list[str],
    figures_dir:   Path,
) -> None:
    gb_model = fitted_models.get("Gradient Boosting")
    if gb_model is None:
        logger.warning("Gradient Boosting not found — skipping feature importance chart")
        return

    fi_df = (pd.DataFrame({"feature": feature_names,
                            "importance": gb_model.feature_importances_})
               .sort_values("importance", ascending=False)
               .reset_index(drop=True))
    total = fi_df["importance"].sum()
    top20 = fi_df.head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    colors = [BLUE if i % 2 == 0 else "#3a6abf" for i in range(len(top20))]
    bars = ax.barh(top20["feature"][::-1], top20["importance"][::-1],
                   color=colors[::-1], edgecolor="none", height=0.65)
    for bar, imp in zip(bars, top20["importance"][::-1]):
        share = imp / total * 100
        ax.text(imp + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{imp:.4f}  ({share:.1f}%)", va="center", fontsize=8, color=FG)
    ax.set_xlabel("Importance (mean Gini impurity reduction)")
    ax.set_title("Top 20 Predictive Features — Gradient Boosting",
                 color=FG, fontsize=13, fontweight="bold")
    for sp in ax.spines.values(): sp.set_edgecolor(DIV)
    ax.grid(axis="x", color=DIV, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    _save("07_feature_importance.png", figures_dir)


def plot_model_comparison(
    results_df:  pd.DataFrame,
    cv_results:  dict[str, dict],
    figures_dir: Path,
) -> None:
    names      = results_df["Model"].tolist()
    n          = len(names)
    cv_means   = [cv_results[nm]["test_auc"].mean() for nm in names]
    cv_stds    = [cv_results[nm]["test_auc"].std()  for nm in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Model Performance Comparison", color=FG, fontsize=13, fontweight="bold", y=1.01)

    bars = axes[0].bar(names, cv_means, color=PAL[:n], edgecolor="none", alpha=0.9, width=0.5)
    axes[0].errorbar(names, cv_means, yerr=cv_stds,
                     fmt="none", color=FG, capsize=6, linewidth=1.5)
    for bar, v, s in zip(bars, cv_means, cv_stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + s + 0.004,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9, color=FG, fontweight="bold")
    axes[0].set_facecolor(CARD)
    axes[0].set_title("CV AUC (5-fold mean ± std)", color=FG, fontsize=11, fontweight="bold")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_ylim([0.72, max(cv_means) + max(cv_stds)*2.5])
    axes[0].tick_params(axis="x", rotation=12)
    for sp in axes[0].spines.values(): sp.set_edgecolor(DIV)
    axes[0].grid(axis="y", color=DIV, linewidth=0.5)

    metrics_show = ["Test AUC","Avg Prec","Precision","Recall","F1"]
    x_pos  = np.arange(len(metrics_show))
    width  = 0.18
    for i, row in results_df.iterrows():
        vals   = [row[m] for m in metrics_show]
        offset = (i - n/2 + 0.5) * width
        axes[1].bar(x_pos + offset, vals, width, label=row["Model"],
                    color=PAL[i], edgecolor="none", alpha=0.88)
    axes[1].set_facecolor(CARD)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics_show, rotation=10)
    axes[1].set_title("Test Set Metrics — All Models", color=FG, fontsize=11, fontweight="bold")
    axes[1].set_ylim([0.45, 1.00])
    axes[1].set_ylabel("Score")
    axes[1].legend(facecolor=CARD, edgecolor=DIV, labelcolor=FG, fontsize=8.5, loc="lower left")
    for sp in axes[1].spines.values(): sp.set_edgecolor(DIV)
    axes[1].grid(axis="y", color=DIV, linewidth=0.5)

    plt.tight_layout()
    _save("08_model_comparison.png", figures_dir)


def plot_all(
    df_raw:        pd.DataFrame,
    y_test:        np.ndarray,
    test_probas:   dict[str, np.ndarray],
    test_preds:    dict[str, np.ndarray],
    fitted_models: dict[str, Any],
    feature_names: list[str],
    cv_results:    dict[str, dict],
    results_df:    pd.DataFrame,
    best_name:     str,
    figures_dir:   str | Path,
) -> None:
    """Generate and save all evaluation charts."""
    figures_dir = Path(figures_dir)
    best_proba  = test_probas[best_name]
    best_preds_ = test_preds[best_name]
    overall_churn_pct = (df_raw["Churn"] == "Yes").mean() * 100

    # ── 01 churn overview ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Customer Churn — Overview", color=FG, fontsize=14, fontweight="bold", y=1.02)
    counts = df_raw["Churn"].value_counts()
    bars = axes[0].bar(["Retained","Churned"], counts.values,
                       color=[GREEN,RED], edgecolor=DIV, linewidth=0.5, width=0.45)
    for bar, (_, v) in zip(bars, counts.items()):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
                     f"{v:,}\n({v/len(df_raw)*100:.1f}%)", ha="center", va="bottom",
                     fontsize=10, color=FG, fontweight="bold")
    axes[0].set_facecolor(CARD); axes[0].set_title("Class Distribution", color=FG, fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Customers"); axes[0].set_ylim([0, counts.max()*1.25])
    for sp in axes[0].spines.values(): sp.set_edgecolor(DIV)
    axes[0].grid(axis="y", color=DIV, linewidth=0.5)
    ct = df_raw.groupby("Contract")["Churn"].apply(lambda x:(x=="Yes").mean()*100).sort_values(ascending=False)
    b2 = axes[1].bar(ct.index, ct.values,
                     color=[RED if v>overall_churn_pct else BLUE for v in ct.values],
                     edgecolor=DIV, linewidth=0.4, width=0.45)
    for bar, v in zip(b2, ct.values):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.6, f"{v:.1f}%",
                     ha="center", va="bottom", fontsize=10, color=FG, fontweight="bold")
    axes[1].axhline(overall_churn_pct, color=GOLD, linestyle="--", linewidth=1.3,
                    label=f"Overall: {overall_churn_pct:.1f}%")
    axes[1].set_facecolor(CARD); axes[1].set_title("Churn by Contract", color=FG, fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)"); axes[1].set_ylim([0,75])
    axes[1].legend(facecolor=CARD, edgecolor=DIV, labelcolor=FG, fontsize=9)
    for sp in axes[1].spines.values(): sp.set_edgecolor(DIV)
    axes[1].grid(axis="y", color=DIV, linewidth=0.5)
    pm = df_raw.groupby("PaymentMethod")["Churn"].apply(lambda x:(x=="Yes").mean()*100).sort_values()
    labels_pm = [p.replace(" (automatic)"," (auto)") for p in pm.index]
    b3 = axes[2].barh(labels_pm, pm.values,
                      color=[RED if v>overall_churn_pct else BLUE for v in pm.values],
                      edgecolor=DIV, linewidth=0.4, height=0.5)
    for bar, v in zip(b3, pm.values):
        axes[2].text(v+0.5, bar.get_y()+bar.get_height()/2, f"{v:.1f}%",
                     va="center", fontsize=9, color=FG, fontweight="bold")
    axes[2].axvline(overall_churn_pct, color=GOLD, linestyle="--", linewidth=1.3,
                    label=f"Overall: {overall_churn_pct:.1f}%")
    axes[2].set_facecolor(CARD); axes[2].set_title("Churn by Payment", color=FG, fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Churn Rate (%)"); axes[2].set_xlim([0,58])
    axes[2].legend(facecolor=CARD, edgecolor=DIV, labelcolor=FG, fontsize=9)
    for sp in axes[2].spines.values(): sp.set_edgecolor(DIV)
    axes[2].grid(axis="x", color=DIV, linewidth=0.5)
    plt.tight_layout(); _save("01_churn_overview.png", figures_dir)

    # ── 02 numeric distributions ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Numeric Feature Distributions by Churn Status",
                 color=FG, fontsize=13, fontweight="bold", y=1.02)
    for ax, (col, title) in zip(axes, [("tenure","Tenure (months)"),
                                        ("MonthlyCharges","Monthly Charges ($)"),
                                        ("TotalCharges","Total Charges ($)")]):
        df_num = df_raw.copy()
        df_num["TotalCharges"] = pd.to_numeric(df_num["TotalCharges"], errors="coerce").fillna(0)
        for label, clr, alpha in [("No",BLUE,0.52),("Yes",RED,0.72)]:
            data = df_num[df_num["Churn"]==label][col]
            mu   = data.mean()
            ax.hist(data, bins=35, color=clr, alpha=alpha, edgecolor="none",
                    label=f"Churn={label}  (n={len(data):,})", density=True)
            ax.axvline(mu, color=clr, linewidth=1.8, linestyle="--",
                       label=f"Mean ({label}): {mu:.0f}")
        ax.set_facecolor(CARD); ax.set_title(title, color=FG, fontsize=11, fontweight="bold")
        ax.set_xlabel(title); ax.set_ylabel("Density")
        ax.legend(facecolor=CARD, edgecolor=DIV, labelcolor=FG, fontsize=8.5)
        for sp in ax.spines.values(): sp.set_edgecolor(DIV)
        ax.grid(axis="y", color=DIV, linewidth=0.5)
    plt.tight_layout(); _save("02_distributions.png", figures_dir)

    # ── 03 categorical breakdown ───────────────────────────────────────────────
    cat_cols = [("gender","Gender"),("SeniorCitizen","Senior Citizen"),
                ("Partner","Has Partner"),("Dependents","Has Dependents"),
                ("PhoneService","Phone Service"),("InternetService","Internet Service")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Churn Rate Across Key Categorical Features",
                 color=FG, fontsize=13, fontweight="bold", y=1.01)
    for ax, (col, title) in zip(axes.flatten(), cat_cols):
        ct2 = df_raw.groupby(col)["Churn"].apply(lambda x:(x=="Yes").mean()*100).sort_values(ascending=False)
        bars = ax.bar(ct2.index.astype(str), ct2.values,
                      color=[RED if v>overall_churn_pct else BLUE for v in ct2.values],
                      edgecolor=DIV, linewidth=0.4, width=0.5)
        for bar, v in zip(bars, ct2.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9.5, color=FG, fontweight="bold")
        ax.axhline(overall_churn_pct, color=GOLD, linestyle="--", linewidth=1.2,
                   label=f"Avg: {overall_churn_pct:.1f}%")
        ax.set_facecolor(CARD); ax.set_title(f"Churn by {title}", color=FG, fontsize=10, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)"); ax.set_ylim([0, ct2.values.max()*1.35])
        ax.legend(facecolor=CARD, edgecolor=DIV, labelcolor=FG, fontsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(DIV)
        ax.grid(axis="y", color=DIV, linewidth=0.5)
    plt.tight_layout(); _save("03_categorical_breakdown.png", figures_dir)

    # ── 04 services count ─────────────────────────────────────────────────────
    svc_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]
    df_svc = df_raw.copy()
    df_svc["svc_count"] = sum((df_svc[c]=="Yes").astype(int) for c in svc_cols)
    sc = df_svc.groupby("svc_count")["Churn"].apply(lambda x:(x=="Yes").mean()*100).reset_index()
    sc.columns = ["svc_count","churn_rate"]
    cust_counts = df_svc["svc_count"].value_counts().sort_index()
    fig, ax1 = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG); ax1.set_facecolor(CARD)
    ax2 = ax1.twinx()
    import matplotlib
    norm_v = (sc["churn_rate"]-sc["churn_rate"].min())/(sc["churn_rate"].max()-sc["churn_rate"].min()+1e-9)
    bar_c  = [matplotlib.colormaps["RdYlGn_r"](v) for v in norm_v]
    ax1.bar(sc["svc_count"], sc["churn_rate"], color=bar_c, edgecolor=DIV, linewidth=0.4, width=0.6, zorder=2)
    ax2.plot(cust_counts.index, cust_counts.values, color=GOLD, linewidth=2.2,
             marker="o", markersize=7, zorder=3, label="Customer count")
    for _, row in sc.iterrows():
        cnt = cust_counts.get(row["svc_count"], 0)
        ax1.text(row["svc_count"], row["churn_rate"]+0.6,
                 f"{row['churn_rate']:.1f}%\n(n={cnt:,})", ha="center", va="bottom",
                 color=FG, fontsize=9, fontweight="bold")
    ax1.set_xlabel("Add-On Services Subscribed"); ax1.set_ylabel("Churn Rate (%)", color=BLUE)
    ax2.set_ylabel("Customer Count", color=GOLD)
    ax1.set_title("Churn Rate and Volume by Services Subscribed", color=FG, fontsize=12, fontweight="bold")
    ax1.set_xticks(sc["svc_count"]); ax1.set_ylim([0, sc["churn_rate"].max()*1.4])
    for sp in ax1.spines.values(): sp.set_edgecolor(DIV)
    ax2.spines["right"].set_edgecolor(DIV)
    ax1.grid(axis="y", color=DIV, linewidth=0.5, zorder=1)
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2, labels2, facecolor=CARD, edgecolor=DIV, labelcolor=FG, fontsize=9)
    plt.tight_layout(); _save("04_services_churn.png", figures_dir)

    # ── 05-08 model eval charts ───────────────────────────────────────────────
    plot_roc_pr(y_test, test_probas, figures_dir)
    plot_confusion_and_metrics(y_test, best_preds_, best_proba, best_name, figures_dir)
    plot_feature_importance(fitted_models, feature_names, figures_dir)
    plot_model_comparison(results_df, cv_results, figures_dir)

    logger.info("All figures saved to %s", figures_dir)
