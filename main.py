# main.py
# ------------------------------------------------------------
# ExtraTrees tuning bundle (MULTI-RUN):
# - Keeps your current best approach:
#   * stable release idx (peak wrist 3D speed)
#   * same feature engineering (stats + fingers + edges + angles)
#   * separate models per target
#   * per-participant models + global fallback
# - Runs a small preset hyperparameter bundle and writes ONE submission per config.
#
# Usage:
#   python main.py
# Then upload the generated files: my_submission__CFG_*.csv
# ------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

# -----------------------------
# 1) CONFIGURATION
# -----------------------------
BASE_PATH = "./"
TRAIN_PATH = f"{BASE_PATH}train.csv"
TEST_PATH = f"{BASE_PATH}test.csv"

SCALER_BOUNDS = {
    "angle": {"min": 30, "max": 60},
    "depth": {"min": -12, "max": 30},
    "left_right": {"min": -16, "max": 16},
}

HALF = 8
FINGERS = ["first", "second", "third", "fourth", "fifth"]

EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("neck", "mid_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("mid_hip", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("mid_hip", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

ANGLE_TRIPLETS = [
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
]

# -----------------------------
# 2) TUNING BUNDLE (EDIT THIS)
# -----------------------------
# These configs are chosen to explore the "high impact" knobs:
# - max_features: controls feature subsampling (very important in high-dim)
# - max_depth: limits depth to reduce overfit
# - min_samples_leaf: regularization
# - bootstrap/max_samples: bagging style that often improves generalization
#
# Start by running all of these once. Then narrow down around the best.
CFG_BUNDLE = [
    dict(name="CFG_f045_d14_leaf1", max_features=0.45, min_samples_leaf=1, max_depth=14, bootstrap=False, max_samples=None)
]

# Core fixed ET params
N_EST = 1400
SEED_ANGLE = 42
SEED_DEPTH = 43
SEED_LR = 44

# -----------------------------
# 3) DATA LOADING
# -----------------------------
def parse_array_column(x):
    if isinstance(x, str):
        s = x.strip()
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            s = s[1:-1]
        return np.fromstring(s, sep=",", dtype=float)
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float).ravel()
    return np.array([], dtype=float)

def load_and_parse(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    exclude_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right", "Unnamed: 0"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Found {len(feature_cols)} feature columns.")
    print("Parsing timeseries data...")
    for col in feature_cols:
        if len(df) > 0 and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(parse_array_column)
    return df, feature_cols

# -----------------------------
# 4) UTILS
# -----------------------------
def _interp_nans(arr):
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        return arr
    if np.all(np.isnan(arr)) or (not np.any(np.isnan(arr))):
        return arr
    idx = np.arange(arr.size)
    m = ~np.isnan(arr)
    out = arr.copy()
    out[~m] = np.interp(idx[~m], idx[m], out[m])
    return out

def _window(arr, center, half=HALF):
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        return arr
    c = int(np.clip(center, 0, arr.size - 1))
    lo = max(0, c - half)
    hi = min(arr.size, c + half + 1)
    return arr[lo:hi]

def _stats(arr):
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        return [0.0] * 10

    nan_frac = float(np.mean(np.isnan(arr)))
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        return [0.0] * 10

    mean = float(v.mean())
    std = float(v.std())
    mn = float(v.min())
    mx = float(v.max())
    first = float(v[0])
    last = float(v[-1])
    slope = float((last - first) / max(v.size - 1, 1))

    filled = _interp_nans(arr)
    if filled.size > 1:
        d = np.diff(filled)
        mad = float(np.mean(np.abs(d)))
        dmax = float(np.max(np.abs(d)))
    else:
        mad, dmax = 0.0, 0.0

    return [mean, std, mn, mx, first, last, slope, mad, dmax, nan_frac]

def _speed_mag(x, y, z):
    x = _interp_nans(x); y = _interp_nans(y); z = _interp_nans(z)
    if x.size < 2:
        return np.array([], dtype=float)
    vx, vy, vz = np.diff(x), np.diff(y), np.diff(z)
    return np.sqrt(vx*vx + vy*vy + vz*vz)

def _peak_speed(x, y, z):
    sp = _speed_mag(x, y, z)
    return float(np.max(sp)) if sp.size else 0.0

def _release_index(x, y, z):
    sp = _speed_mag(x, y, z)
    if sp.size == 0:
        return 0
    return int(np.argmax(sp) + 1)

def _get_series(row, col, default_len=0):
    v = row.get(col, None)
    if isinstance(v, np.ndarray):
        return v
    if hasattr(v, "__len__") and not isinstance(v, str):
        return np.asarray(v, dtype=float).ravel()
    return np.zeros(default_len, dtype=float) if default_len > 0 else np.array([], dtype=float)

def _dist_series(ax, ay, az, bx, by, bz):
    L = min(len(ax), len(bx), len(ay), len(by), len(az), len(bz))
    if L == 0:
        return np.array([], dtype=float)
    ax, ay, az = _interp_nans(ax[:L]), _interp_nans(ay[:L]), _interp_nans(az[:L])
    bx, by, bz = _interp_nans(bx[:L]), _interp_nans(by[:L]), _interp_nans(bz[:L])
    return np.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)

def _angle_series(ax, ay, az, bx, by, bz, cx, cy, cz):
    L = min(len(ax), len(bx), len(cx), len(ay), len(by), len(cy), len(az), len(bz), len(cz))
    if L == 0:
        return np.array([], dtype=float)

    ax, ay, az = _interp_nans(ax[:L]), _interp_nans(ay[:L]), _interp_nans(az[:L])
    bx, by, bz = _interp_nans(bx[:L]), _interp_nans(by[:L]), _interp_nans(bz[:L])
    cx, cy, cz = _interp_nans(cx[:L]), _interp_nans(cy[:L]), _interp_nans(cz[:L])

    BAx, BAy, BAz = ax - bx, ay - by, az - bz
    BCx, BCy, BCz = cx - bx, cy - by, cz - bz

    num = BAx*BCx + BAy*BCy + BAz*BCz
    den = (np.sqrt(BAx*BAx + BAy*BAy + BAz*BAz) *
           np.sqrt(BCx*BCx + BCy*BCy + BCz*BCz) + 1e-9)
    cos = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def scale_value(value, col_name):
    mini = SCALER_BOUNDS[col_name]["min"]
    maxi = SCALER_BOUNDS[col_name]["max"]
    value = float(np.clip(value, mini, maxi))
    scaled = (value - mini) / (maxi - mini)
    return float(np.clip(scaled, 0, 1))

# -----------------------------
# 5) FEATURE EXTRACTION (same as your best pipeline)
# -----------------------------
def extract_features(df, feature_cols):
    features = []

    def is_shooting_focus_col(col, side):
        if not col.startswith(side + "_"):
            return False
        return ("wrist" in col) or ("elbow" in col) or ("shoulder" in col) or ("finger" in col)

    for _, row in df.iterrows():
        row_feats = []

        # choose shooting hand + release idx
        lwx = _get_series(row, "left_wrist_x")
        lwy = _get_series(row, "left_wrist_y", default_len=len(lwx))
        lwz = _get_series(row, "left_wrist_z", default_len=len(lwx))

        rwx = _get_series(row, "right_wrist_x")
        rwy = _get_series(row, "right_wrist_y", default_len=len(rwx))
        rwz = _get_series(row, "right_wrist_z", default_len=len(rwx))

        lpk = _peak_speed(lwx, lwy, lwz)
        rpk = _peak_speed(rwx, rwy, rwz)

        if rpk >= lpk:
            side = "right"
            rel = _release_index(rwx, rwy, rwz)
            wx, wy, wz = rwx, rwy, rwz
        else:
            side = "left"
            rel = _release_index(lwx, lwy, lwz)
            wx, wy, wz = lwx, lwy, lwz

        row_feats.append(1.0 if side == "right" else 0.0)

        # per-column stats (windowed on shooting-side focus)
        for col in feature_cols:
            ts = row[col]
            arr = np.asarray(ts, dtype=float).ravel() if hasattr(ts, "__len__") else np.array([], dtype=float)
            arr_use = _window(arr, rel, HALF) if is_shooting_focus_col(col, side) else arr
            row_feats.extend(_stats(arr_use))

        # finger tip speed
        tip_speeds = []
        for f in FINGERS:
            dx = _get_series(row, f"{side}_{f}_finger_distal_x")
            dy = _get_series(row, f"{side}_{f}_finger_distal_y", default_len=len(dx))
            dz = _get_series(row, f"{side}_{f}_finger_distal_z", default_len=len(dx))

            L = min(len(dx), len(wx), len(dy), len(wy), len(dz), len(wz))
            if L == 0:
                tip_speeds.append(np.nan)
                continue

            rx = _window(dx[:L] - wx[:L], rel, HALF)
            ry = _window(dy[:L] - wy[:L], rel, HALF)
            rz = _window(dz[:L] - wz[:L], rel, HALF)

            sp = _speed_mag(rx, ry, rz)
            tip_speeds.append(float(np.max(sp)) if sp.size else np.nan)

        if np.any(~np.isnan(tip_speeds)):
            row_feats.append(float(np.nanmax(tip_speeds)))
            row_feats.append(float(np.nanmean(tip_speeds)))
        else:
            row_feats.extend([0.0, 0.0])

        # finger spread distances
        tdx = _get_series(row, f"{side}_first_finger_distal_x")
        tdy = _get_series(row, f"{side}_first_finger_distal_y", default_len=len(tdx))
        tdz = _get_series(row, f"{side}_first_finger_distal_z", default_len=len(tdx))

        idx_x = _get_series(row, f"{side}_second_finger_distal_x")
        idx_y = _get_series(row, f"{side}_second_finger_distal_y", default_len=len(idx_x))
        idx_z = _get_series(row, f"{side}_second_finger_distal_z", default_len=len(idx_x))

        pkx = _get_series(row, f"{side}_fifth_finger_distal_x")
        pky = _get_series(row, f"{side}_fifth_finger_distal_y", default_len=len(pkx))
        pkz = _get_series(row, f"{side}_fifth_finger_distal_z", default_len=len(pkx))

        d_thumb_index = _dist_series(_window(tdx, rel), _window(tdy, rel), _window(tdz, rel),
                                     _window(idx_x, rel), _window(idx_y, rel), _window(idx_z, rel))
        d_index_pinky = _dist_series(_window(idx_x, rel), _window(idx_y, rel), _window(idx_z, rel),
                                     _window(pkx, rel), _window(pky, rel), _window(pkz, rel))
        row_feats.extend(_stats(d_thumb_index))
        row_feats.extend(_stats(d_index_pinky))

        # finger curl angles: fingers 2-5
        for f in ["second", "third", "fourth", "fifth"]:
            mcp_x = _get_series(row, f"{side}_{f}_finger_mcp_x")
            mcp_y = _get_series(row, f"{side}_{f}_finger_mcp_y", default_len=len(mcp_x))
            mcp_z = _get_series(row, f"{side}_{f}_finger_mcp_z", default_len=len(mcp_x))

            pip_x = _get_series(row, f"{side}_{f}_finger_pip_x")
            pip_y = _get_series(row, f"{side}_{f}_finger_pip_y", default_len=len(pip_x))
            pip_z = _get_series(row, f"{side}_{f}_finger_pip_z", default_len=len(pip_x))

            dip_x = _get_series(row, f"{side}_{f}_finger_dip_x")
            dip_y = _get_series(row, f"{side}_{f}_finger_dip_y", default_len=len(dip_x))
            dip_z = _get_series(row, f"{side}_{f}_finger_dip_z", default_len=len(dip_x))

            dis_x = _get_series(row, f"{side}_{f}_finger_distal_x")
            dis_y = _get_series(row, f"{side}_{f}_finger_distal_y", default_len=len(dis_x))
            dis_z = _get_series(row, f"{side}_{f}_finger_distal_z", default_len=len(dis_x))

            ang_pip = _angle_series(_window(mcp_x, rel), _window(mcp_y, rel), _window(mcp_z, rel),
                                    _window(pip_x, rel), _window(pip_y, rel), _window(pip_z, rel),
                                    _window(dip_x, rel), _window(dip_y, rel), _window(dip_z, rel))
            ang_dip = _angle_series(_window(pip_x, rel), _window(pip_y, rel), _window(pip_z, rel),
                                    _window(dip_x, rel), _window(dip_y, rel), _window(dip_z, rel),
                                    _window(dis_x, rel), _window(dis_y, rel), _window(dis_z, rel))
            row_feats.extend(_stats(ang_pip))
            row_feats.extend(_stats(ang_dip))

        # thumb angles
        cmc_x = _get_series(row, f"{side}_first_finger_cmc_x")
        cmc_y = _get_series(row, f"{side}_first_finger_cmc_y", default_len=len(cmc_x))
        cmc_z = _get_series(row, f"{side}_first_finger_cmc_z", default_len=len(cmc_x))

        mcp_x = _get_series(row, f"{side}_first_finger_mcp_x")
        mcp_y = _get_series(row, f"{side}_first_finger_mcp_y", default_len=len(mcp_x))
        mcp_z = _get_series(row, f"{side}_first_finger_mcp_z", default_len=len(mcp_x))

        ip_x = _get_series(row, f"{side}_first_finger_ip_x")
        ip_y = _get_series(row, f"{side}_first_finger_ip_y", default_len=len(ip_x))
        ip_z = _get_series(row, f"{side}_first_finger_ip_z", default_len=len(ip_x))

        dis_x = _get_series(row, f"{side}_first_finger_distal_x")
        dis_y = _get_series(row, f"{side}_first_finger_distal_y", default_len=len(dis_x))
        dis_z = _get_series(row, f"{side}_first_finger_distal_z", default_len=len(dis_x))

        ang_thumb_mcp = _angle_series(_window(cmc_x, rel), _window(cmc_y, rel), _window(cmc_z, rel),
                                      _window(mcp_x, rel), _window(mcp_y, rel), _window(mcp_z, rel),
                                      _window(ip_x, rel), _window(ip_y, rel), _window(ip_z, rel))
        ang_thumb_ip = _angle_series(_window(mcp_x, rel), _window(mcp_y, rel), _window(mcp_z, rel),
                                     _window(ip_x, rel), _window(ip_y, rel), _window(ip_z, rel),
                                     _window(dis_x, rel), _window(dis_y, rel), _window(dis_z, rel))
        row_feats.extend(_stats(ang_thumb_mcp))
        row_feats.extend(_stats(ang_thumb_ip))

        # skeletal distances
        for a, b in EDGES:
            ax = _get_series(row, f"{a}_x")
            ay = _get_series(row, f"{a}_y", default_len=len(ax))
            az = _get_series(row, f"{a}_z", default_len=len(ax))

            bx = _get_series(row, f"{b}_x")
            by = _get_series(row, f"{b}_y", default_len=len(bx))
            bz = _get_series(row, f"{b}_z", default_len=len(bx))

            d = _dist_series(_window(ax, rel), _window(ay, rel), _window(az, rel),
                             _window(bx, rel), _window(by, rel), _window(bz, rel))
            row_feats.extend(_stats(d))

        # joint angles
        for A, B, C in ANGLE_TRIPLETS:
            Ax = _get_series(row, f"{A}_x")
            Ay = _get_series(row, f"{A}_y", default_len=len(Ax))
            Az = _get_series(row, f"{A}_z", default_len=len(Ax))

            Bx = _get_series(row, f"{B}_x")
            By = _get_series(row, f"{B}_y", default_len=len(Bx))
            Bz = _get_series(row, f"{B}_z", default_len=len(Bx))

            Cx = _get_series(row, f"{C}_x")
            Cy = _get_series(row, f"{C}_y", default_len=len(Cx))
            Cz = _get_series(row, f"{C}_z", default_len=len(Cx))

            ang = _angle_series(_window(Ax, rel), _window(Ay, rel), _window(Az, rel),
                                _window(Bx, rel), _window(By, rel), _window(Bz, rel),
                                _window(Cx, rel), _window(Cy, rel), _window(Cz, rel))
            row_feats.extend(_stats(ang))

        features.append(row_feats)

    X = np.asarray(features, dtype=float)
    print(f"Extracted features shape: {X.shape}")
    return X

# -----------------------------
# 6) TRAIN / PREDICT (single config)
# -----------------------------
def make_et(cfg, seed):
    # note: max_samples only valid if bootstrap=True in sklearn
    kwargs = dict(
        n_estimators=N_EST,
        random_state=seed,
        n_jobs=-1,
        max_features=cfg["max_features"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_depth=cfg["max_depth"],
        bootstrap=cfg["bootstrap"],
    )
    if cfg["bootstrap"] and (cfg["max_samples"] is not None):
        kwargs["max_samples"] = cfg["max_samples"]
    return ExtraTreesRegressor(**kwargs)

def train_and_predict_one_cfg(cfg, train_df, test_df, X_train, X_test):
    y_angle = train_df["angle"].values
    y_depth = train_df["depth"].values
    y_lr    = train_df["left_right"].values

    # global fallback (separate per target)
    g_angle = make_et(cfg, SEED_ANGLE).fit(X_train, y_angle)
    g_depth = make_et(cfg, SEED_DEPTH).fit(X_train, y_depth)
    g_lr    = make_et(cfg, SEED_LR).fit(X_train, y_lr)

    # per participant packs
    packs = {}
    for pid in sorted(train_df["participant_id"].unique()):
        idx = (train_df["participant_id"].values == pid)
        pid = int(pid)
        packs[pid] = (
            make_et(cfg, SEED_ANGLE).fit(X_train[idx], y_angle[idx]),
            make_et(cfg, SEED_DEPTH).fit(X_train[idx], y_depth[idx]),
            make_et(cfg, SEED_LR).fit(X_train[idx], y_lr[idx]),
        )

    # predict
    preds = np.zeros((len(test_df), 3), dtype=float)
    for i, pid in enumerate(test_df["participant_id"].values):
        pid = int(pid)
        x = X_test[i:i+1]
        if pid in packs:
            ma, md, ml = packs[pid]
        else:
            ma, md, ml = g_angle, g_depth, g_lr
        preds[i, 0] = ma.predict(x)[0]
        preds[i, 1] = md.predict(x)[0]
        preds[i, 2] = ml.predict(x)[0]

    return preds

def write_submission(preds, test_df, out_path):
    sub = pd.DataFrame()
    sub["id"] = test_df["id"]
    sub["scaled_angle"] = [scale_value(v, "angle") for v in preds[:, 0]]
    sub["scaled_depth"] = [scale_value(v, "depth") for v in preds[:, 1]]
    sub["scaled_left_right"] = [scale_value(v, "left_right") for v in preds[:, 2]]
    sub.to_csv(out_path, index=False)
    return sub

# -----------------------------
# 7) MAIN: run bundle
# -----------------------------
def main():
    train_df, feature_cols = load_and_parse(TRAIN_PATH)
    test_df, _ = load_and_parse(TEST_PATH)

    print("Extracting features...")
    X_train = extract_features(train_df, feature_cols)
    X_test = extract_features(test_df, feature_cols)

    print("\nRunning ExtraTrees tuning bundle...")
    for cfg in CFG_BUNDLE:
        name = cfg["name"]
        out_path = f"my_submission__{name}.csv"
        print(f"\n=== {name} ===")
        print({k: cfg[k] for k in cfg if k != "name"})

        preds = train_and_predict_one_cfg(cfg, train_df, test_df, X_train, X_test)
        sub = write_submission(preds, test_df, out_path)

        print(f"Wrote: {out_path}")
        print(sub.head(2))

    print("\nDone. Upload the generated my_submission__CFG_*.csv files and keep the best LB score.")

if __name__ == "__main__":
    main()
