"""
================================================================
MEMBER 1 — COMPLETE ANALYSIS & REPORTING SCRIPT
Industrial Predictive Maintenance System
================================================================
Run AFTER main.py has saved processed_train.csv & processed_test.csv

What this script produces:
  plots/1_class_distribution.png
  plots/2_correlation_heatmap.png
  plots/3_sensor_signals.png
  plots/4_missing_values.png
  plots/5_outlier_detection.png
  plots/6_rolling_features.png
  plots/7_lag_features.png
  plots/8_roc_features.png
  plots/9_fft_features.png
  plots/10_feature_importance.png
  plots/11_drift_report.png
  reports/drift_report.csv
  processed/processed_train_full.csv
  processed/processed_test_full.csv
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.fft import fft, fftfreq
from scipy.stats import ks_2samp, kurtosis, skew, zscore
from scipy.spatial.distance import jensenshannon

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────
DATA_DIR     = r"D:\my_ml_project\data"
PROCESSED    = r"D:\my_ml_project\processed"
PLOTS_DIR    = r"D:\my_ml_project\plots"
REPORTS_DIR  = r"D:\my_ml_project\reports"

for d in [PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_CSV = os.path.join(PROCESSED, "processed_train.csv")
TEST_CSV  = os.path.join(PROCESSED, "processed_test.csv")

# Sensor column names (no header in raw CSVs — assigned by main.py)
COL_NAMES = ["current", "vib_x", "vib_y", "vib_z",
             "volt_a",  "volt_b", "volt_c", "volt_d"]

# ─────────────────────────────────────────────────────────────
# HELPER: load one raw CSV (for signal visualisation)
# ─────────────────────────────────────────────────────────────
def load_raw_sample(data_path, folder, filename):
    path = os.path.join(data_path, folder, filename)
    df   = pd.read_csv(path, header=None, names=COL_NAMES)
    return df.apply(pd.to_numeric, errors="coerce")


# ================================================================
# SECTION 1 — DATASET UNDERSTANDING
# ================================================================

def section1_dataset_understanding(train_df):
    print("\n" + "="*60)
    print("  SECTION 1 — DATASET UNDERSTANDING")
    print("="*60)

    print(f"""
DATASET STRUCTURE
─────────────────
Source   : Fault Induction Motor Dataset (Kaggle)
Format   : CSV files with NO header row — 8 raw sensor columns
Structure:
  data/
  ├── normal/          (49 recordings  → label 0 — healthy motor)
  └── imbalance/       (333 recordings → label 1 — faulty motor)
      ├── 6g, 10g, 15g, 20g, 25g, 30g, 35g  (fault severity)

8 SENSOR CHANNELS
  Col 0 — current   : Motor phase current (A)
  Col 1 — vib_x     : Vibration X-axis (g)
  Col 2 — vib_y     : Vibration Y-axis (g)
  Col 3 — vib_z     : Vibration Z-axis (g)
  Col 4 — volt_a    : Voltage channel A (V)
  Col 5 — volt_b    : Voltage channel B (V)
  Col 6 — volt_c    : Voltage channel C (V)
  Col 7 — volt_d    : Voltage channel D (V)

LABELS  : Binary — 0 = Normal, 1 = Imbalance Fault
TEMPORAL: Each CSV is one sequential recording session (~300k samples @ high freq)
""")

    print(f"Processed train shape : {train_df.shape}")
    print(f"Features              : {train_df.shape[1]-1}")
    print(f"Label distribution:\n{train_df['fault_label'].value_counts().to_string()}")

    print("\nDescriptive Statistics (first 8 features):")
    print(train_df.iloc[:, :8].describe().round(3).to_string())


# ================================================================
# SECTION 2 — MISSING VALUES REPORT
# ================================================================

def section2_missing_values(data_path):
    print("\n" + "="*60)
    print("  SECTION 2 — MISSING VALUE ANALYSIS")
    print("="*60)

    # Load one normal and one fault file to demonstrate
    methods   = ["Forward Fill", "Interpolation", "Median Imputation"]
    miss_data = {}

    for folder, fname in [("normal", "12.288.csv"),
                          ("imbalance/10g", "13.9264.csv")]:
        fpath = os.path.join(data_path, folder, fname)
        if not os.path.exists(fpath):
            continue
        raw = pd.read_csv(fpath, header=None, names=COL_NAMES)
        raw = raw.apply(pd.to_numeric, errors="coerce")

        # Artificially introduce 5% missing values for demonstration
        demo = raw.copy()
        mask = np.random.random(demo.shape) < 0.05
        demo[mask] = np.nan
        miss_data[folder.split("/")[0]] = demo

    if not miss_data:
        print("  Raw files not found — skipping missing value demo.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Missing Value Handling — Method Comparison", fontsize=13, fontweight="bold")

    for row_idx, (fname, demo) in enumerate(miss_data.items()):
        col_demo = "current"
        original = demo[col_demo].copy()

        results = {
            "Forward Fill"      : original.ffill().bfill(),
            "Interpolation"     : original.interpolate(method="linear").bfill(),
            "Median Imputation" : original.fillna(original.median()),
        }

        for col_idx, (method, filled) in enumerate(results.items()):
            ax = axes[row_idx][col_idx]
            ax.plot(filled.values[:200], color="#3498db", linewidth=0.8, label="Filled")
            nan_idx = original.index[original.isna()][:200]
            ax.scatter(nan_idx[nan_idx < 200], filled.iloc[nan_idx[nan_idx < 200]],
                       color="#e74c3c", s=20, zorder=5, label="Was NaN")
            ax.set_title(f"{fname} — {method}", fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Current (A)")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "4_missing_values.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")
    print("\nChosen Method: INTERPOLATION")
    print("  Reason: Linear interpolation respects the temporal trend between")
    print("  adjacent sensor readings. Forward fill ignores future context.")
    print("  Median imputation ignores time structure entirely.")


# ================================================================
# SECTION 3 — CLASS DISTRIBUTION PLOT
# ================================================================

def section3_class_distribution(train_df):
    print("\n" + "="*60)
    print("  SECTION 3 — CLASS DISTRIBUTION")
    print("="*60)

    counts = train_df["fault_label"].value_counts().sort_index()
    labels = {0: "Normal", 1: "Imbalance Fault"}
    colors = ["#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution — Fault Labels", fontsize=13, fontweight="bold")

    bars = axes[0].bar([labels[i] for i in counts.index],
                       counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Absolute Count")
    axes[0].set_ylabel("Number of Recordings")
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 2, str(v),
                     ha="center", fontweight="bold", fontsize=11)

    axes[1].pie(counts.values, labels=[labels[i] for i in counts.index],
                autopct="%1.1f%%", colors=colors, startangle=140,
                pctdistance=0.75, textprops={"fontsize": 11})
    axes[1].set_title("Proportion")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "1_class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")
    print(f"\n  Normal    : {counts.get(0, 0)} recordings ({counts.get(0,0)/len(train_df)*100:.1f}%)")
    print(f"  Fault     : {counts.get(1, 0)} recordings ({counts.get(1,0)/len(train_df)*100:.1f}%)")
    print(f"  Imbalance ratio: {counts.get(1,0)/counts.get(0,1):.1f}:1 (fault:normal)")


# ================================================================
# SECTION 4 — CORRELATION HEATMAP
# ================================================================

def section4_correlation_heatmap(train_df):
    print("\n" + "="*60)
    print("  SECTION 4 — CORRELATION HEATMAP")
    print("="*60)

    # Use mean features for heatmap (one per sensor)
    mean_cols = [c for c in train_df.columns if c.endswith("_mean")]
    if not mean_cols:
        mean_cols = train_df.columns[:8].tolist()

    corr = train_df[mean_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Sensor Feature Correlation Heatmap\n(Mean features per sensor channel)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "2_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# ================================================================
# SECTION 5 — SENSOR SIGNAL VISUALIZATION
# ================================================================

def section5_sensor_signals(data_path):
    print("\n" + "="*60)
    print("  SECTION 5 — SENSOR SIGNAL VISUALIZATION")
    print("="*60)

    normal_path = os.path.join(data_path, "normal", "12.288.csv")
    fault_path  = os.path.join(data_path, "imbalance", "10g", "13.9264.csv")

    if not os.path.exists(normal_path) or not os.path.exists(fault_path):
        print("  Raw files not found — skipping signal visualization.")
        return

    normal_df = pd.read_csv(normal_path, header=None, names=COL_NAMES,
                             nrows=500).apply(pd.to_numeric, errors="coerce")
    fault_df  = pd.read_csv(fault_path,  header=None, names=COL_NAMES,
                             nrows=500).apply(pd.to_numeric, errors="coerce")

    plot_cols = ["current", "vib_x", "vib_y", "volt_a"]
    colors    = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]

    fig, axes = plt.subplots(len(plot_cols), 2, figsize=(16, 10), sharex=True)
    fig.suptitle("Sensor Signal Comparison: Normal vs Fault (first 500 samples)",
                 fontsize=13, fontweight="bold")

    for i, (col, color) in enumerate(zip(plot_cols, colors)):
        axes[i][0].plot(normal_df[col].values, color=color, linewidth=0.8)
        axes[i][0].set_ylabel(col, fontsize=9)
        axes[i][0].grid(True, alpha=0.3)
        if i == 0:
            axes[i][0].set_title("NORMAL", fontsize=11, fontweight="bold", color="#2ecc71")

        axes[i][1].plot(fault_df[col].values, color="#e74c3c", linewidth=0.8)
        axes[i][1].grid(True, alpha=0.3)
        if i == 0:
            axes[i][1].set_title("FAULT (Imbalance)", fontsize=11,
                                  fontweight="bold", color="#e74c3c")

    axes[-1][0].set_xlabel("Sample Index")
    axes[-1][1].set_xlabel("Sample Index")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "3_sensor_signals.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# ================================================================
# SECTION 6 — OUTLIER DETECTION
# ================================================================

def section6_outlier_detection(train_df):
    print("\n" + "="*60)
    print("  SECTION 6 — OUTLIER DETECTION (IQR + Z-SCORE)")
    print("="*60)

    mean_cols = [c for c in train_df.columns if "_mean" in c][:4]
    if not mean_cols:
        mean_cols = train_df.columns[:4].tolist()

    fig, axes = plt.subplots(2, len(mean_cols), figsize=(16, 8))
    fig.suptitle("Outlier Detection — IQR vs Z-Score", fontsize=13, fontweight="bold")

    iqr_counts  = {}
    zsco_counts = {}

    for i, col in enumerate(mean_cols):
        vals = train_df[col].dropna()

        # IQR method
        Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        iqr_out = (vals < lo) | (vals > hi)
        iqr_counts[col] = iqr_out.sum()

        axes[0][i].scatter(range(len(vals)), vals, c=iqr_out.map({True:"#e74c3c", False:"#3498db"}),
                           s=8, alpha=0.6)
        axes[0][i].axhline(lo, color="orange", linestyle="--", linewidth=1, label=f"IQR bounds")
        axes[0][i].axhline(hi, color="orange", linestyle="--", linewidth=1)
        axes[0][i].set_title(f"{col}\nIQR outliers: {iqr_out.sum()}", fontsize=9)
        axes[0][i].set_ylabel("Value")
        axes[0][i].grid(True, alpha=0.3)

        # Z-score method
        z      = np.abs(zscore(vals))
        z_out  = z > 3
        zsco_counts[col] = z_out.sum()

        axes[1][i].scatter(range(len(vals)), vals,
                           c=pd.Series(z_out).map({True:"#e74c3c", False:"#2ecc71"}),
                           s=8, alpha=0.6)
        axes[1][i].set_title(f"Z-score outliers: {z_out.sum()}", fontsize=9)
        axes[1][i].set_ylabel("Value")
        axes[1][i].set_xlabel("Sample Index")
        axes[1][i].grid(True, alpha=0.3)

    axes[0][0].set_ylabel("IQR Method")
    axes[1][0].set_ylabel("Z-Score Method")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "5_outlier_detection.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")
    print("\n  IQR outlier counts  :", iqr_counts)
    print("  Z-score outlier counts:", zsco_counts)
    print("\n  Chosen strategy: CLIP to IQR bounds (preserves temporal continuity)")


# ================================================================
# SECTION 7 — ADVANCED FEATURE ENGINEERING (on raw sample)
# ================================================================

def section7_feature_engineering(data_path):
    print("\n" + "="*60)
    print("  SECTION 7 — ADVANCED FEATURE ENGINEERING")
    print("="*60)

    # Load one raw file for demonstration
    fpath = os.path.join(data_path, "imbalance", "10g", "13.9264.csv")
    if not os.path.exists(fpath):
        print("  Raw file not found — skipping feature engineering demo plots.")
        return

    raw = pd.read_csv(fpath, header=None, names=COL_NAMES,
                      nrows=300).apply(pd.to_numeric, errors="coerce")
    raw = raw.ffill().bfill()

    # ── A. Rolling Window Features ────────────────────────────
    print("\n  A. Rolling Window Features")
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Rolling Window Features — vib_x (windows: 5, 10, 20)",
                 fontsize=12, fontweight="bold")

    axes[0].plot(raw["vib_x"], color="#95a5a6", linewidth=0.7, label="Raw signal")
    for w, color in zip([5, 10, 20], ["#e74c3c", "#3498db", "#2ecc71"]):
        axes[0].plot(raw["vib_x"].rolling(w).mean(), color=color,
                     linewidth=1.2, label=f"roll_mean_{w}")
    axes[0].set_ylabel("Roll Mean")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for w, color in zip([5, 10, 20], ["#e74c3c", "#3498db", "#2ecc71"]):
        axes[1].plot(raw["vib_x"].rolling(w).std(), color=color,
                     linewidth=1.2, label=f"roll_std_{w}")
    axes[1].set_ylabel("Roll Std")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    for w, color in zip([5, 10, 20], ["#e74c3c", "#3498db", "#2ecc71"]):
        roll_min = raw["vib_x"].rolling(w).min()
        roll_max = raw["vib_x"].rolling(w).max()
        axes[2].fill_between(range(len(raw)), roll_min, roll_max,
                             alpha=0.25, color=color, label=f"range_{w}")
    axes[2].set_ylabel("Roll Min/Max Range")
    axes[2].set_xlabel("Sample")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "6_rolling_features.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {out}")

    # ── B. Lag Features ───────────────────────────────────────
    print("  B. Lag Features")
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle("Lag Features — current channel\n"
                 "(Lag features expose temporal history — rising lag trend = early bearing wear)",
                 fontsize=11, fontweight="bold")

    for ax, lag in zip(axes.flat, [1, 2, 5, 10]):
        lagged = raw["current"].shift(lag)
        ax.scatter(raw["current"], lagged, s=4, alpha=0.4, color="#3498db")
        ax.set_xlabel("current (t)")
        ax.set_ylabel(f"current (t-{lag})")
        ax.set_title(f"Lag {lag}")
        ax.grid(True, alpha=0.3)
        corr = raw["current"].corr(lagged)
        ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=9, color="#e74c3c")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "7_lag_features.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {out}")

    # ── C. Rate of Change Features ────────────────────────────
    print("  C. Rate of Change Features")
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Rate of Change Features — current channel",
                 fontsize=12, fontweight="bold")

    diff    = raw["current"].diff()
    pct_chg = raw["current"].pct_change().replace([np.inf, -np.inf], 0)
    slope   = raw["current"].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)

    axes[0].plot(diff,    color="#e74c3c",  linewidth=0.8)
    axes[0].set_ylabel("Difference (diff)")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pct_chg, color="#f39c12",  linewidth=0.8)
    axes[1].set_ylabel("% Change (pct_change)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(slope,   color="#2ecc71",  linewidth=0.8)
    axes[2].set_ylabel("Slope (5-sample window)")
    axes[2].set_xlabel("Sample")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "8_roc_features.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {out}")

    # ── D. FFT Frequency Domain Features ──────────────────────
    print("  D. FFT Frequency Domain Features")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("FFT Frequency Domain Analysis\n"
                 "(Bearing faults → sidebands at BPFO/BPFI | Imbalance → 1× RPM peak)",
                 fontsize=11, fontweight="bold")

    for ax, col in zip(axes.flat, ["vib_x", "vib_y", "current", "volt_a"]):
        segment  = raw[col].dropna().values[:256]
        spectrum = np.abs(fft(segment))[:128]
        freqs    = fftfreq(256)[:128]

        dom_freq  = abs(freqs[np.argmax(spectrum)])
        sp_energy = np.sum(spectrum**2)
        psd       = spectrum**2
        psd_n     = psd / (psd.sum() + 1e-10)
        sp_ent    = -np.sum(psd_n * np.log(psd_n + 1e-10))

        ax.plot(freqs, spectrum, color="#9b59b6", linewidth=0.8)
        ax.fill_between(freqs, spectrum, alpha=0.2, color="#9b59b6")
        ax.set_title(f"{col}", fontsize=10)
        ax.set_xlabel("Frequency (normalized)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        info = f"Dom freq: {dom_freq:.3f}\nEnergy: {sp_energy:.1f}\nEntropy: {sp_ent:.2f}"
        ax.text(0.58, 0.72, info, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "9_fft_features.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {out}")


# ================================================================
# SECTION 8 — FEATURE IMPORTANCE & SELECTION
# ================================================================

def section8_feature_selection(train_df):
    print("\n" + "="*60)
    print("  SECTION 8 — FEATURE SELECTION")
    print("="*60)

    feat_cols = [c for c in train_df.columns if c != "fault_label"]
    X = train_df[feat_cols].fillna(0).values
    y = train_df["fault_label"].values

    # Mutual Information
    print("  Computing Mutual Information scores...")
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=feat_cols).sort_values(ascending=False)

    # Random Forest Importance
    print("  Training Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)

    print(f"\n  Top 5 by Mutual Information:")
    print(mi_series.head(5).to_string())
    print(f"\n  Top 5 by Random Forest:")
    print(rf_imp.head(5).to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Feature Selection — Mutual Information vs Random Forest Importance",
                 fontsize=12, fontweight="bold")

    mi_series.head(20).sort_values().plot(
        kind="barh", ax=axes[0], color="#3498db", edgecolor="white")
    axes[0].set_title("Top 20 — Mutual Information", fontsize=11)
    axes[0].set_xlabel("MI Score")

    rf_imp.head(20).sort_values().plot(
        kind="barh", ax=axes[1], color="#e74c3c", edgecolor="white")
    axes[1].set_title("Top 20 — Random Forest Importance", fontsize=11)
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "10_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Saved: {out}")

    return mi_series, rf_imp


# ================================================================
# SECTION 9 — SCALING COMPARISON
# ================================================================

def section9_scaling(train_df):
    print("\n" + "="*60)
    print("  SECTION 9 — FEATURE SCALING COMPARISON")
    print("="*60)

    feat_cols = [c for c in train_df.columns if c != "fault_label"][:6]
    X = train_df[feat_cols].fillna(0)

    std_sc  = StandardScaler()
    mm_sc   = MinMaxScaler()
    X_std   = pd.DataFrame(std_sc.fit_transform(X),  columns=feat_cols)
    X_mm    = pd.DataFrame(mm_sc.fit_transform(X),   columns=feat_cols)

    print("""
  StandardScaler  : zero mean, unit variance
    → Best for: SVM, PCA, logistic regression, k-NN
    → Robust to fault outliers vs MinMaxScaler
    → CHOSEN for this project

  MinMaxScaler    : maps to [0, 1]
    → Best for: neural networks with sigmoid activation
    → Sensitive to extreme outlier values (fault spikes shift Xmax)
    → NOT recommended for sensor fault data

  Why scaling matters:
    → Distance-based models (k-NN, SVM) treat large-range features
      as more important — unscaled volt_a (230V) dominates vib_x (0.02g)
    → Gradient descent converges faster on scaled inputs
    → Regularisation penalties apply equally across all features
""")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Feature Scaling Comparison", fontsize=12, fontweight="bold")

    X[feat_cols[0]].hist(ax=axes[0], bins=30, color="#95a5a6")
    axes[0].set_title("Raw (unscaled)")

    X_std[feat_cols[0]].hist(ax=axes[1], bins=30, color="#3498db")
    axes[1].set_title("StandardScaler (μ=0, σ=1)")

    X_mm[feat_cols[0]].hist(ax=axes[2], bins=30, color="#2ecc71")
    axes[2].set_title("MinMaxScaler [0, 1]")

    for ax in axes:
        ax.set_xlabel(feat_cols[0])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "0_scaling_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {out}")


# ================================================================
# SECTION 10 — FULL DRIFT MONITORING (KS + PSI + JS)
# ================================================================

def compute_psi(base, current, bins=10):
    """Population Stability Index."""
    base_cnt, edges = np.histogram(base, bins=bins)
    curr_cnt, _     = np.histogram(current, bins=edges)
    bp = (base_cnt + 1e-6) / (len(base) + 1e-6 * bins)
    cp = (curr_cnt + 1e-6) / (len(current) + 1e-6 * bins)
    return float(np.sum((cp - bp) * np.log(cp / bp)))


def section10_drift_monitoring(train_df, test_df):
    print("\n" + "="*60)
    print("  SECTION 10 — DRIFT MONITORING (KS + PSI + JS)")
    print("="*60)

    print("""
  DATA DRIFT DEFINITION
  ─────────────────────
  Data drift occurs when the statistical distribution of production
  sensor readings P(X_production) diverges from training data P(X_train).

  WHY IT MATTERS IN INDUSTRIAL SYSTEMS
  ─────────────────────────────────────
  • Seasonal temperature changes → bearing lubrication viscosity shifts
  • Mechanical wear → normal operating signature changes over months
  • Sensor calibration drift → baseline readings shift gradually
  • New product run on same machine → load profile changes

  IMPACT ON MODEL RELIABILITY
  ────────────────────────────
  A model trained on P_train(X) uses decision boundaries learned in a
  different feature space. If P_production(X) differs significantly,
  the model applies wrong boundaries → missed failures (low recall).
""")

    feat_cols = [c for c in train_df.columns if c != "fault_label"]
    report    = []

    for col in feat_cols:
        if col not in test_df.columns:
            continue
        base    = train_df[col].dropna().values
        current = test_df[col].dropna().values
        if len(base) < 5 or len(current) < 5:
            continue

        ks_stat, p_val = ks_2samp(base, current)
        psi_val        = compute_psi(base, current)

        # JS divergence
        rng = (min(base.min(), current.min()), max(base.max(), current.max()))
        bh, _ = np.histogram(base,    bins=20, range=rng, density=True)
        ch, _ = np.histogram(current, bins=20, range=rng, density=True)
        bh = bh / (bh.sum() + 1e-10)
        ch = ch / (ch.sum() + 1e-10)
        js  = float(jensenshannon(bh, ch))

        drift = p_val < 0.05
        psi_status = ("Major"    if psi_val > 0.2 else
                      "Moderate" if psi_val > 0.1 else "Stable")

        report.append({
            "Feature"       : col,
            "KS_Stat"       : round(ks_stat, 4),
            "P_Value"       : round(p_val,   4),
            "PSI"           : round(psi_val, 4),
            "JS_Divergence" : round(js,      4),
            "Drift_Alert"   : "YES" if drift else "NO",
            "PSI_Status"    : psi_status,
        })

    report_df = pd.DataFrame(report)
    out_csv   = os.path.join(REPORTS_DIR, "drift_report.csv")
    report_df.to_csv(out_csv, index=False)
    print(f"  ✅ Drift report saved: {out_csv}")

    n_drift = (report_df["Drift_Alert"] == "YES").sum()
    print(f"\n  Features checked : {len(report_df)}")
    print(f"  Drifting features: {n_drift}")
    print(f"\n  Top drifting features:")
    print(report_df.sort_values("KS_Stat", ascending=False).head(10).to_string(index=False))

    # Visualise top 4 drifting or all features
    drift_feats = report_df[report_df["Drift_Alert"] == "YES"]["Feature"].tolist()
    plot_feats  = drift_feats[:4] if drift_feats else report_df["Feature"].tolist()[:4]

    if plot_feats:
        fig, axes = plt.subplots(1, len(plot_feats), figsize=(5*len(plot_feats), 4))
        if len(plot_feats) == 1:
            axes = [axes]
        fig.suptitle("Sensor Distribution: Training vs Test (Drift Check)",
                     fontsize=12, fontweight="bold")
        for ax, col in zip(axes, plot_feats):
            ax.hist(train_df[col].dropna(), bins=25, alpha=0.6,
                    color="#3498db", label="Train", density=True)
            ax.hist(test_df[col].dropna(),  bins=25, alpha=0.6,
                    color="#e74c3c", label="Test",  density=True)
            ax.set_title(col, fontsize=9)
            ax.legend(fontsize=8)
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, "11_drift_report.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved: {out}")

    return report_df


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  MEMBER 1 — COMPLETE ANALYSIS SCRIPT")
    print("█"*60)

    # Load processed files saved by main.py
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"\nERROR: Processed files not found.")
        print(f"  Expected: {TRAIN_CSV}")
        print(f"  Run main.py first, then re-run this script.")
        exit(1)

    print(f"\nLoading processed data from {PROCESSED}...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"  Train: {train_df.shape}  |  Test: {test_df.shape}")

    # Run all sections
    section1_dataset_understanding(train_df)
    section2_missing_values(DATA_DIR)
    section3_class_distribution(train_df)
    section4_correlation_heatmap(train_df)
    section5_sensor_signals(DATA_DIR)
    section6_outlier_detection(train_df)
    section7_feature_engineering(DATA_DIR)
    mi_scores, rf_scores = section8_feature_selection(train_df)
    section9_scaling(train_df)
    drift_report = section10_drift_monitoring(train_df, test_df)

    print("\n" + "█"*60)
    print("  ✅  ALL SECTIONS COMPLETE")
    print("█"*60)
    print(f"""
  Generated files:
  ┌─ plots/
  │   ├── 0_scaling_comparison.png
  │   ├── 1_class_distribution.png
  │   ├── 2_correlation_heatmap.png
  │   ├── 3_sensor_signals.png
  │   ├── 4_missing_values.png
  │   ├── 5_outlier_detection.png
  │   ├── 6_rolling_features.png
  │   ├── 7_lag_features.png
  │   ├── 8_roc_features.png
  │   ├── 9_fft_features.png
  │   ├── 10_feature_importance.png
  │   └── 11_drift_report.png
  └─ reports/
      └── drift_report.csv
""")