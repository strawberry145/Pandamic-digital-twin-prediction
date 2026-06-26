"""Pandemic Digital Twin Visualization Pipeline

Generates comprehensive PNG visualizations for pandemic risk prediction:
- Model performance metrics
- Outbreak status distribution
- Infection risk heatmap
- Region-level case forecasts
- Time-series predictions

Usage:
  python pandemic_digital_twin_viz.py
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from pandemic_digital_twin import (
    load_data, feature_engineering, get_individual_features,
    build_region_aggregates, get_region_features,
    train_classifier, train_regressor,
    TARGET_LABEL_MAPS, NUMERIC_FEATURES, ORDINAL_FEATURES, NOMINAL_FEATURES
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
#  STYLE
# ─────────────────────────────────────────────
DARK_BG   = "#0a0e1a"
PANEL_BG  = "#0d1428"
GRID_COL  = "#1a2440"
CYAN      = "#00e5ff"
GREEN     = "#39ff14"
AMBER     = "#ffb800"
RED       = "#ff3366"
PURPLE    = "#a855f7"
BLUE      = "#3b82f6"
TEXT_PRI  = "#c8d6e8"
TEXT_SEC  = "#6b7f99"

CMAP_RISK = LinearSegmentedColormap.from_list(
    "risk", [GREEN, AMBER, RED]
)

def style():
    plt.rcParams.update({
        "figure.facecolor":     DARK_BG,
        "axes.facecolor":       PANEL_BG,
        "axes.edgecolor":       GRID_COL,
        "axes.labelcolor":      TEXT_PRI,
        "axes.titlecolor":      CYAN,
        "axes.titlesize":       11,
        "axes.labelsize":       9,
        "axes.grid":            True,
        "grid.color":           GRID_COL,
        "grid.linewidth":       0.5,
        "xtick.color":          TEXT_SEC,
        "ytick.color":          TEXT_SEC,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.facecolor":     PANEL_BG,
        "legend.edgecolor":     GRID_COL,
        "legend.labelcolor":    TEXT_PRI,
        "legend.fontsize":      8,
        "text.color":           TEXT_PRI,
        "font.family":          "monospace",
        "lines.linewidth":      1.5,
        "figure.dpi":           130,
    })

style()

def panel_title(ax, txt):
    ax.set_title(txt, color=CYAN, fontsize=10, fontweight="bold",
                 loc="left", pad=8, fontfamily="monospace")

def add_watermark(fig):
    fig.text(0.99, 0.01, "PANDEMIC DIGITAL TWIN | EARLY WARNING SYSTEM",
             ha="right", va="bottom", color=TEXT_SEC, fontsize=7,
             fontfamily="monospace", alpha=0.5)

# ─────────────────────────────────────────────
#  FIGURE 1 — MODEL PERFORMANCE & METRICS
# ─────────────────────────────────────────────
def fig_model_performance(outbreak_model, risk_model, cases_model,
                         X_outbreak_test, y_outbreak_test, y_outbreak_pred,
                         X_risk_test, y_risk_test, y_risk_pred,
                         y_cases_test, y_cases_pred):
    print("[Fig 1] Model Performance & Metrics...")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("PANDEMIC MODELS — PERFORMANCE EVALUATION",
                 color=CYAN, fontsize=14, fontweight="bold",
                 fontfamily="monospace", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    # 1. Outbreak confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm_outbreak = confusion_matrix(y_outbreak_test, y_outbreak_pred)
    sns.heatmap(cm_outbreak, annot=True, fmt="d", ax=ax1,
                cmap=LinearSegmentedColormap.from_list("cm", [PANEL_BG, CYAN]),
                linewidths=0.5, linecolor=DARK_BG,
                annot_kws={"size": 10, "color": "white"},
                cbar=False, cbar_kws={"shrink": 0.7})
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
    ax1.set_xticklabels(["No", "Emerging", "Ongoing"], fontsize=8)
    ax1.set_yticklabels(["No", "Emerging", "Ongoing"], fontsize=8, rotation=0)
    panel_title(ax1, "Outbreak Confusion Matrix")

    # 2. Risk confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm_risk = confusion_matrix(y_risk_test, y_risk_pred)
    sns.heatmap(cm_risk, annot=True, fmt="d", ax=ax2,
                cmap=LinearSegmentedColormap.from_list("cm", [PANEL_BG, RED]),
                linewidths=0.5, linecolor=DARK_BG,
                annot_kws={"size": 10, "color": "white"},
                cbar=False)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.set_xticklabels(["Low", "Med", "High"], fontsize=8)
    ax2.set_yticklabels(["Low", "Med", "High"], fontsize=8, rotation=0)
    panel_title(ax2, "Risk Confusion Matrix")

    # 3. Cases actual vs predicted
    ax3 = fig.add_subplot(gs[0, 2])
    lim_lo = min(y_cases_test.min(), y_cases_pred.min())
    lim_hi = max(y_cases_test.max(), y_cases_pred.max())
    ax3.scatter(y_cases_test, y_cases_pred, alpha=0.15, s=6, color=AMBER, rasterized=True)
    ax3.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=GREEN, lw=1.5, ls="--")
    ax3.set_xlabel("Actual Cases")
    ax3.set_ylabel("Predicted Cases")
    panel_title(ax3, "Daily Cases — Actual vs Pred")

    # 4. Outbreak accuracy by class
    ax4 = fig.add_subplot(gs[1, 0])
    outbreak_labels = ["No Outbreak", "Emerging", "Ongoing"]
    outbreak_accs = []
    for i in range(3):
        mask = y_outbreak_test == i
        if mask.sum() > 0:
            acc = (y_outbreak_pred[mask] == i).mean()
        else:
            acc = 0
        outbreak_accs.append(acc)
    ax4.bar(outbreak_labels, outbreak_accs, color=[GREEN, AMBER, RED], alpha=0.8, edgecolor="none")
    ax4.set_ylim(0, 1.1)
    ax4.axhline(1.0, color=GRID_COL, lw=0.8, ls="--")
    for i, v in enumerate(outbreak_accs):
        ax4.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=8, color=TEXT_PRI)
    panel_title(ax4, "Outbreak Class Recall")
    ax4.set_ylabel("Recall")

    # 5. Risk accuracy by class
    ax5 = fig.add_subplot(gs[1, 1])
    risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
    risk_accs = []
    for i in range(3):
        mask = y_risk_test == i
        if mask.sum() > 0:
            acc = (y_risk_pred[mask] == i).mean()
        else:
            acc = 0
        risk_accs.append(acc)
    ax5.bar(risk_labels, risk_accs, color=[GREEN, AMBER, RED], alpha=0.8, edgecolor="none")
    ax5.set_ylim(0, 1.1)
    ax5.axhline(1.0, color=GRID_COL, lw=0.8, ls="--")
    for i, v in enumerate(risk_accs):
        ax5.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=8, color=TEXT_PRI)
    panel_title(ax5, "Risk Class Recall")
    ax5.set_ylabel("Recall")

    # 6. Cases residuals
    ax6 = fig.add_subplot(gs[1, 2])
    residuals = y_cases_test - y_cases_pred
    ax6.scatter(y_cases_pred, residuals, alpha=0.15, s=6, color=PURPLE, rasterized=True)
    ax6.axhline(0, color=RED, lw=1.5, ls="--")
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Residual")
    panel_title(ax6, "Cases Residual Plot")

    # 7. Outbreak class distribution
    ax7 = fig.add_subplot(gs[2, 0])
    outbreak_counts = np.bincount(y_outbreak_test, minlength=3)
    ax7.bar(outbreak_labels, outbreak_counts, color=[GREEN, AMBER, RED], alpha=0.7, edgecolor="none")
    for i, v in enumerate(outbreak_counts):
        ax7.text(i, v+20, str(v), ha="center", fontsize=8, color=TEXT_PRI)
    panel_title(ax7, "Outbreak Test Set Distribution")
    ax7.set_ylabel("Count")

    # 8. Risk class distribution
    ax8 = fig.add_subplot(gs[2, 1])
    risk_counts = np.bincount(y_risk_test, minlength=3)
    ax8.bar(risk_labels, risk_counts, color=[GREEN, AMBER, RED], alpha=0.7, edgecolor="none")
    for i, v in enumerate(risk_counts):
        ax8.text(i, v+100, str(v), ha="center", fontsize=8, color=TEXT_PRI)
    panel_title(ax8, "Risk Test Set Distribution")
    ax8.set_ylabel("Count")

    # 9. Summary metrics cards
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    metrics = [
        ("OUTBREAK", "Samples", f"{len(y_outbreak_test):,}", CYAN),
        ("RISK", "Samples", f"{len(y_risk_test):,}", BLUE),
        ("CASES", "MAE", f"{np.mean(np.abs(residuals)):.2f}", GREEN),
    ]
    for j, (model_name, metric, val, col) in enumerate(metrics):
        y_pos = 0.88 - j * 0.25
        rect = FancyBboxPatch((0.02, y_pos-0.10), 0.96, 0.18,
                               boxstyle="round,pad=0.01",
                               facecolor=DARK_BG, edgecolor=col, linewidth=1)
        ax9.add_patch(rect)
        ax9.text(0.08, y_pos+0.03, model_name, fontsize=8, color=TEXT_SEC,
                 fontfamily="monospace")
        ax9.text(0.08, y_pos-0.03, metric, fontsize=10, color=TEXT_PRI,
                 fontfamily="monospace", fontweight="bold")
        ax9.text(0.88, y_pos, val, fontsize=14, color=col,
                 fontfamily="monospace", fontweight="bold", ha="right", va="center")
    panel_title(ax9, "Summary")

    add_watermark(fig)
    fig.savefig("fig1_model_performance.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=130)
    plt.close()
    print("  Saved: fig1_model_performance.png")


# ─────────────────────────────────────────────
#  FIGURE 2 — PREDICTION DISTRIBUTION
# ─────────────────────────────────────────────
def fig_predictions_dist(df_preds):
    print("[Fig 2] Prediction Distribution...")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("PANDEMIC PREDICTIONS — DISTRIBUTION & INSIGHTS",
                 color=CYAN, fontsize=13, fontweight="bold",
                 fontfamily="monospace", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # 1. Predicted outbreak distribution
    ax1 = fig.add_subplot(gs[0, 0])
    outbreak_preds = df_preds["predicted_outbreak"].value_counts().sort_index()
    labels = ["No Outbreak", "Emerging", "Ongoing"]
    colors = [GREEN, AMBER, RED]
    ax1.bar([labels[i] for i in outbreak_preds.index], outbreak_preds.values,
            color=colors[:len(outbreak_preds)], alpha=0.8, edgecolor="none")
    for i, v in enumerate(outbreak_preds.values):
        ax1.text(i, v+5, str(v), ha="center", fontsize=8, color=TEXT_PRI)
    panel_title(ax1, "Predicted Outbreak Status")
    ax1.set_ylabel("Count")

    # 2. Predicted risk distribution
    ax2 = fig.add_subplot(gs[0, 1])
    risk_preds = df_preds["predicted_risk"].value_counts().sort_index()
    labels_risk = ["Low Risk", "Medium Risk", "High Risk"]
    ax2.bar([labels_risk[i] for i in risk_preds.index], risk_preds.values,
            color=[GREEN, AMBER, RED][:len(risk_preds)], alpha=0.8, edgecolor="none")
    for i, v in enumerate(risk_preds.values):
        ax2.text(i, v+5, str(v), ha="center", fontsize=8, color=TEXT_PRI)
    panel_title(ax2, "Predicted Infection Risk")
    ax2.set_ylabel("Count")

    # 3. Predicted daily cases distribution
    ax3 = fig.add_subplot(gs[0, 2])
    cases = df_preds["predicted_daily_cases"]
    ax3.hist(cases, bins=40, color=CYAN, alpha=0.7, edgecolor=GRID_COL)
    ax3.axvline(cases.mean(), color=GREEN, lw=2, ls="--", label=f"μ={cases.mean():.1f}")
    ax3.axvline(cases.median(), color=AMBER, lw=2, ls=":", label=f"med={cases.median():.1f}")
    panel_title(ax3, "Predicted Daily Cases Distribution")
    ax3.set_xlabel("Cases")
    ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=7)

    # 4. Risk by outbreak status
    ax4 = fig.add_subplot(gs[1, 0])
    risk_by_outbreak = pd.crosstab(df_preds["predicted_outbreak"], df_preds["predicted_risk"])
    risk_by_outbreak.index = ["No Outbreak", "Emerging", "Ongoing"]
    risk_by_outbreak.columns = ["Low", "Medium", "High"]
    sns.heatmap(risk_by_outbreak, annot=True, fmt="d", ax=ax4,
                cmap=LinearSegmentedColormap.from_list("heat", [DARK_BG, RED]),
                cbar=False, linewidths=0.5, linecolor=GRID_COL)
    panel_title(ax4, "Risk vs Outbreak Status")
    ax4.set_xlabel("Risk Level")
    ax4.set_ylabel("Outbreak Status")

    # 5. Cases by predicted risk
    ax5 = fig.add_subplot(gs[1, 1])
    risk_cat = df_preds["predicted_risk"].map({0: "Low", 1: "Medium", 2: "High"})
    data_by_risk = [df_preds[risk_cat == cat]["predicted_daily_cases"].values
                    for cat in ["Low", "Medium", "High"]]
    bp = ax5.boxplot(data_by_risk, patch_artist=True, widths=0.6,
                     labels=["Low", "Medium", "High"],
                     medianprops={"color": CYAN, "lw": 2},
                     whiskerprops={"color": TEXT_SEC},
                     capprops={"color": TEXT_SEC})
    for patch, col in zip(bp["boxes"], [GREEN, AMBER, RED]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    panel_title(ax5, "Cases Distribution by Risk")
    ax5.set_ylabel("Predicted Daily Cases")

    # 6. Top predictors
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    panel_title(ax6, "Key Insights")
    insights = [
        f"Total samples: {len(df_preds):,}",
        f"Avg predicted cases: {cases.mean():.1f}",
        f"High-risk count: {(df_preds['predicted_risk']==2).sum():,}",
        f"Ongoing outbreak: {(df_preds['predicted_outbreak']==2).sum():,}",
    ]
    for j, text in enumerate(insights):
        ax6.text(0.1, 0.85 - j*0.20, text, fontsize=9, color=TEXT_PRI,
                 fontfamily="monospace", transform=ax6.transAxes)

    add_watermark(fig)
    fig.savefig("fig2_predictions_distribution.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=130)
    plt.close()
    print("  Saved: fig2_predictions_distribution.png")


# ─────────────────────────────────────────────
#  FIGURE 3 — REGION RISK HEATMAP
# ─────────────────────────────────────────────
def fig_region_heatmap(df_preds):
    print("[Fig 3] Region Risk Heatmap...")
    if "Location" not in df_preds.columns:
        print("  Skipping (no Location in predictions)")
        return

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("PANDEMIC RISK BY REGION",
                 color=CYAN, fontsize=13, fontweight="bold",
                 fontfamily="monospace", y=0.98)
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    location_risk = pd.crosstab(df_preds["Location"], df_preds["predicted_risk"])
    location_risk.columns = ["Low", "Medium", "High"]
    sns.heatmap(location_risk, annot=True, fmt="d", ax=ax,
                cmap=CMAP_RISK, cbar_kws={"label": "Risk Count"},
                linewidths=0.5, linecolor=GRID_COL)
    panel_title(ax, "Risk Predictions by Location")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Location")

    add_watermark(fig)
    fig.savefig("fig3_region_heatmap.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=130)
    plt.close()
    print("  Saved: fig3_region_heatmap.png")


# ─────────────────────────────────────────────
#  FIGURE 4 — OUTBREAK TIMELINE
# ─────────────────────────────────────────────
def fig_outbreak_timeline(df):
    print("[Fig 4] Outbreak Timeline...")
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("OUTBREAK STATUS — TEMPORAL DISTRIBUTION",
                 color=CYAN, fontsize=13, fontweight="bold",
                 fontfamily="monospace", y=0.98)
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    df_ts = df.copy()
    df_ts["Date"] = pd.to_datetime(df_ts["Date_of_Data_Collection"])
    daily_outbreak = df_ts.groupby(["Date", "Outbreak_Status"]).size().unstack(fill_value=0)

    if "No Outbreak" in daily_outbreak.columns:
        ax.plot(daily_outbreak.index, daily_outbreak["No Outbreak"],
                color=GREEN, lw=2, label="No Outbreak", marker="o", markersize=3)
    if "Emerging Outbreak" in daily_outbreak.columns:
        ax.plot(daily_outbreak.index, daily_outbreak["Emerging Outbreak"],
                color=AMBER, lw=2, label="Emerging Outbreak", marker="s", markersize=3)
    if "Ongoing Outbreak" in daily_outbreak.columns:
        ax.plot(daily_outbreak.index, daily_outbreak["Ongoing Outbreak"],
                color=RED, lw=2, label="Ongoing Outbreak", marker="^", markersize=3)

    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    panel_title(ax, "Outbreak Status Over Time")
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=45, fontsize=8)

    add_watermark(fig)
    fig.savefig("fig4_outbreak_timeline.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=130)
    plt.close()
    print("  Saved: fig4_outbreak_timeline.png")


# ─────────────────────────────────────────────
#  FIGURE 5 — CASES TIME-SERIES
# ─────────────────────────────────────────────
def fig_cases_timeline(df):
    print("[Fig 5] Cases Time-Series...")
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("DAILY NEW CASES — TIME SERIES",
                 color=CYAN, fontsize=13, fontweight="bold",
                 fontfamily="monospace", y=0.98)
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    df_ts = df.copy()
    df_ts["Date"] = pd.to_datetime(df_ts["Date_of_Data_Collection"])
    daily_cases = df_ts.groupby("Date")["Daily_New_Cases"].agg(["mean", "min", "max"])

    ax.plot(daily_cases.index, daily_cases["mean"],
            color=CYAN, lw=2, label="Mean", marker="o", markersize=4)
    ax.fill_between(daily_cases.index, daily_cases["min"], daily_cases["max"],
                    color=CYAN, alpha=0.1, label="Min–Max Range")

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily New Cases")
    panel_title(ax, "Daily New Cases Over Time")
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=45, fontsize=8)

    add_watermark(fig)
    fig.savefig("fig5_cases_timeline.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=130)
    plt.close()
    print("  Saved: fig5_cases_timeline.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pandemic Digital Twin Visualization")
    parser.add_argument("--csv", type=str, default="public_health_surveillance_dataset.csv")
    args = parser.parse_args()

    print("=" * 55)
    print(" PANDEMIC DIGITAL TWIN — VISUALIZATION PIPELINE")
    print("=" * 55)

    print("\n[1/3] Loading and preparing data...")
    df = load_data(args.csv)
    df = feature_engineering(df)

    print("\n[2/3] Training models and generating predictions...")
    individual_df = df.dropna(subset=["Outbreak_Status", "Infection_Risk_Level", "Daily_New_Cases"]).copy()
    X_individual = get_individual_features(individual_df)
    y_outbreak = individual_df["Outbreak_Status"].map(TARGET_LABEL_MAPS["Outbreak_Status"])
    y_risk = individual_df["Infection_Risk_Level"].map(TARGET_LABEL_MAPS["Infection_Risk_Level"])
    y_cases = individual_df["Daily_New_Cases"]

    outbreak_model, X_o_test, y_o_test, y_o_pred = train_classifier(X_individual, y_outbreak, "Outbreak_Status")
    risk_model, X_r_test, y_r_test, y_r_pred = train_classifier(X_individual, y_risk, "Infection_Risk_Level")
    cases_model, X_c_test, y_c_test, y_c_pred = train_regressor(X_individual, y_cases, "Daily_New_Cases")

    sample = individual_df.sample(min(500, len(individual_df)), random_state=42)
    df_preds = sample.copy()
    df_preds["predicted_outbreak"] = outbreak_model.predict(get_individual_features(sample))
    df_preds["predicted_risk"] = risk_model.predict(get_individual_features(sample))
    df_preds["predicted_daily_cases"] = cases_model.predict(get_individual_features(sample))

    print("\n[3/3] Generating visualizations...")
    fig_model_performance(outbreak_model, risk_model, cases_model,
                         X_o_test, y_o_test, y_o_pred,
                         X_r_test, y_r_test, y_r_pred,
                         y_c_test, y_c_pred)
    fig_predictions_dist(df_preds)
    fig_region_heatmap(df_preds)
    fig_outbreak_timeline(df)
    fig_cases_timeline(df)

    print("\nDone!")
    print("-" * 55)
    outputs = [
        "fig1_model_performance.png",
        "fig2_predictions_distribution.png",
        "fig3_region_heatmap.png",
        "fig4_outbreak_timeline.png",
        "fig5_cases_timeline.png",
    ]
    for f in outputs:
        size = os.path.getsize(f) // 1024 if os.path.exists(f) else 0
        print(f"  ✓  {f}  ({size} KB)")
    print("=" * 55)


if __name__ == "__main__":
    main()
