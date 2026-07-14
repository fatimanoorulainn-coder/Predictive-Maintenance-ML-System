import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, kurtosis, skew
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("NOTE: pip install imbalanced-learn for SMOTE. Using fallback.")

DATA_DIR  = r"D:\my_ml_project\data"
OUT_DIR   = r"D:\my_ml_project\processed"

# Column names assigned manually — 8 sensor channels per the dataset
# (current + 7 vibration/voltage axes, no header in raw CSVs)
COL_NAMES = ["current", "vib_x", "vib_y", "vib_z", "volt_a", "volt_b", "volt_c", "volt_d"]


# ==========================================
# STEP 1: SUMMARIZE EACH FILE → ONE ROW
# ==========================================

def summarize_file(file_path):
    """
    Read one headerless CSV, assign column names, then collapse
    the entire recording into one row of statistical features.
    """
    df = pd.read_csv(file_path, header=None, names=COL_NAMES)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(method="linear").bfill().ffill()

    row = {}
    for col in COL_NAMES:
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue

        rms = np.sqrt(np.mean(vals ** 2))

        row[f"{col}_mean"]          = float(np.mean(vals))
        row[f"{col}_std"]           = float(np.std(vals))
        row[f"{col}_min"]           = float(np.min(vals))
        row[f"{col}_max"]           = float(np.max(vals))
        row[f"{col}_rms"]           = float(rms)
        row[f"{col}_p2p"]           = float(np.max(vals) - np.min(vals))
        row[f"{col}_kurt"]          = float(kurtosis(vals))
        row[f"{col}_skew"]          = float(skew(vals))
        row[f"{col}_crest"]         = float(np.max(np.abs(vals)) / rms) if rms != 0 else 0.0

        if len(vals) >= 100:
            row[f"{col}_roll_std"]  = float(pd.Series(vals).rolling(100).std().dropna().mean())
        else:
            row[f"{col}_roll_std"]  = row[f"{col}_std"]

    return row


# ==========================================
# STEP 2: LOAD ALL FILES
# ==========================================

def load_and_preprocess(data_path):
    records = []
    loaded  = 0

    print("Summarizing recordings (one row per file)...")

    for root, dirs, files in os.walk(data_path):
        for filename in sorted(files):
            if not filename.lower().endswith(".csv"):
                continue

            file_path  = os.path.join(root, filename)
            relative   = os.path.relpath(root, data_path)
            top_folder = relative.split(os.sep)[0].lower()

            if top_folder == "normal":
                label = 0
            elif top_folder == "imbalance":
                label = 1
            else:
                continue

            try:
                row = summarize_file(file_path)
            except Exception as e:
                print(f"\n  WARNING: skipping {filename} → {e}")
                continue

            row["fault_label"] = label
            records.append(row)
            loaded += 1
            print(f"  [{loaded}] {relative}\\{filename}", end="\r")

    print()

    df = pd.DataFrame(records).reset_index(drop=True)
    df = df.fillna(df.median(numeric_only=True))

    print(f"\n✅ Dataset ready : {df.shape[0]} recordings × {df.shape[1]-1} features")
    print(f"   Remaining NaNs : {df.isnull().sum().sum()}")
    print(f"   Label distribution: {Counter(df['fault_label'])}")
    return df


# ==========================================
# STEP 3: DRIFT MONITORING
# ==========================================

def monitor_drift(train_df, live_df, feature_cols):
    print("\n--- Drift Monitoring Report (KS Test) ---")
    drifted = []
    for col in feature_cols[:10]:
        stat, p_val = ks_2samp(train_df[col].dropna(), live_df[col].dropna())
        status = "⚠️  DRIFT" if p_val < 0.05 else "✅ Stable"
        print(f"  {col:35s}  p={p_val:.4f}  {status}")
        if p_val < 0.05:
            drifted.append(col)
    if drifted:
        print(f"\n  ALERT: {len(drifted)} feature(s) drifting — consider retraining.")
    else:
        print("\n  All monitored features stable.")


# ==========================================
# STEP 4: SMOTE FALLBACK
# ==========================================

def simple_oversample(X, y):
    counts = Counter(y)
    maj_n  = max(counts.values())
    Xl, yl = [X], [y]
    for cls, cnt in counts.items():
        if cnt < maj_n:
            X_cls    = X[y == cls]
            n_needed = maj_n - cnt
            idx      = np.random.choice(len(X_cls), n_needed, replace=True)
            Xl.append(X_cls[idx])
            yl.append(np.full(n_needed, cls))
    return np.vstack(Xl), np.concatenate(yl)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # A. Load
    df = load_and_preprocess(DATA_DIR)

    feature_names = [c for c in df.columns if c != "fault_label"]
    X = df[feature_names].values
    y = df["fault_label"].values

    # B. Impute safety net
    imputer  = SimpleImputer(strategy="median")
    X        = imputer.fit_transform(X)
    print(f"Feature matrix : {X.shape}  |  NaNs: {np.isnan(X).sum()}")

    # C. Scale
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # D. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # E. Balance training set only
    n_minority = Counter(y_train)[0]
    k = min(5, n_minority - 1)

    if IMBLEARN_AVAILABLE and k >= 1:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        print(f"Resampled (SMOTE)    : {Counter(y_train_bal)}")
    else:
        X_train_bal, y_train_bal = simple_oversample(X_train, y_train)
        print(f"Resampled (fallback) : {Counter(y_train_bal)}")

    # F. Drift monitoring
    monitor_drift(
        pd.DataFrame(X_train, columns=feature_names),
        pd.DataFrame(X_test,  columns=feature_names),
        feature_names
    )

    # G. Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)

    train_out = pd.DataFrame(X_train_bal, columns=feature_names)
    train_out["fault_label"] = y_train_bal
    train_out.to_csv(os.path.join(OUT_DIR, "processed_train.csv"), index=False)

    test_out = pd.DataFrame(X_test, columns=feature_names)
    test_out["fault_label"] = y_test
    test_out.to_csv(os.path.join(OUT_DIR, "processed_test.csv"), index=False)

    print(f"\n✅ SUCCESS — Member 1 pipeline complete.")
    print(f"   {OUT_DIR}\\processed_train.csv  → Member 2")
    print(f"   {OUT_DIR}\\processed_test.csv   → Member 3")