
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional imports
_HAS_TF = False
try:
    from tensorflow.keras.models import load_model
    _HAS_TF = True
except Exception:
    pass

_HAS_XGB = False
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    pass

st.set_page_config(page_title="ML Predictor (ANN / RF / XGB)", layout="wide")
st.title("Predictor: ANN · Random Forest · XGBoost")

# ---------------- PATHS (edit in sidebar) ----------------
ART_DIR_ANN = st.sidebar.text_input("ANN artifacts folder", value="artifacts_ann")
ART_DIR_RF  = st.sidebar.text_input("RF artifacts folder",  value="artifacts_rf_v2")
ART_DIR_XGB = st.sidebar.text_input("XGB artifacts folder", value="artifacts_xgb")

FN_XSC = st.sidebar.text_input("X-scaler filename", value="Xsc.pkl")
FN_YSC = st.sidebar.text_input("Y-scaler filename", value="Ysc.pkl")
FN_META = st.sidebar.text_input("Meta filename", value="meta.joblib")

FN_ANN = st.sidebar.text_input("ANN model file (.h5)", value="ann_model.h5")
FN_RF  = st.sidebar.text_input("RF model file (.joblib)", value="rf_model.joblib")
FN_XGB = st.sidebar.text_input("XGB model file (.joblib or .json)", value="xgb_model.joblib")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: move your files into the folders above, or edit the paths here.")

def _safe(path1, path2): return os.path.join(path1, path2)

def load_artifacts(folder):
    out = {"meta": None, "xsc": None, "ysc": None}
    try: out["meta"] = joblib.load(_safe(folder, FN_META))
    except Exception: pass
    try: out["xsc"]  = joblib.load(_safe(folder, FN_XSC))
    except Exception: pass
    try: out["ysc"]  = joblib.load(_safe(folder, FN_YSC))
    except Exception: pass
    return out

def get_cfg(meta):
    cfg = dict(log_transform_X=False, log_transform_Y=False,
               standardize_X=False, standardize_Y=False,
               X_columns=None, Y_columns=None)
    if isinstance(meta, dict):
        for k in list(cfg.keys()):
            if k in meta: cfg[k] = meta[k]
        if cfg["X_columns"] is None and "x_columns" in meta: cfg["X_columns"] = meta["x_columns"]
        if cfg["Y_columns"] is None and "y_columns" in meta: cfg["Y_columns"] = meta["y_columns"]
    return cfg

def fwd_X(dfX, cfg, xsc):
    X = dfX.copy()
    if cfg.get("log_transform_X", False): X = np.log(X + 1e-9)
    if cfg.get("standardize_X", False) and xsc is not None: X = xsc.transform(X)
    else: X = X.values if hasattr(X, "values") else np.asarray(X)
    return X

def inv_Y(Yhat, cfg, ysc):
    Y = Yhat.copy()
    if cfg.get("standardize_Y", False) and ysc is not None: Y = ysc.inverse_transform(Y)
    if cfg.get("log_transform_Y", False): Y = np.exp(Y)
    return Y

# Load artifacts
ann_art = load_artifacts(ART_DIR_ANN)
rf_art  = load_artifacts(ART_DIR_RF)
xgb_art = load_artifacts(ART_DIR_XGB)

ann_cfg = get_cfg(ann_art["meta"]); rf_cfg = get_cfg(rf_art["meta"]); xgb_cfg = get_cfg(xgb_art["meta"])

FEATURES = ann_cfg.get("X_columns") or rf_cfg.get("X_columns") or xgb_cfg.get("X_columns") or []
TARGETS  = ann_cfg.get("Y_columns") or rf_cfg.get("Y_columns") or xgb_cfg.get("Y_columns") or ["Y1"]

if not FEATURES:
    st.warning("Feature names not found in meta files. Provide them below (comma-separated).")
    txt = st.text_input("Feature names", value="")
    if txt:
        FEATURES = [t.strip() for t in txt.split(",") if t.strip()]

# Load models
models = {}
c1,c2,c3 = st.columns(3)
with c1:
    ann_path = _safe(ART_DIR_ANN, FN_ANN)
    if os.path.exists(ann_path) and _HAS_TF:
        try: models["ANN"] = load_model(ann_path); st.success("ANN loaded")
        except Exception as e: st.error(f"ANN load error: {e}")
    else: st.info("ANN file absent or TensorFlow not installed.")

with c2:
    rf_path = _safe(ART_DIR_RF, FN_RF)
    if os.path.exists(rf_path):
        try: models["RF"] = joblib.load(rf_path); st.success("RF loaded")
        except Exception as e: st.error(f"RF load error: {e}")
    else: st.info("RF file absent.")

with c3:
    xgb_path = _safe(ART_DIR_XGB, FN_XGB)
    if os.path.exists(xgb_path):
        try:
            if xgb_path.endswith(".json") and _HAS_XGB:
                booster = xgb.Booster(); booster.load_model(xgb_path)
                st.warning("XGB Booster loaded (batch predict via sklearn API is recommended).")
                models["XGB_raw"] = booster
            else:
                models["XGB"] = joblib.load(xgb_path); st.success("XGB loaded")
        except Exception as e: st.error(f"XGB load error: {e}")
    else: st.info("XGB file absent.")

st.markdown("---")
st.subheader("Single prediction")

cols = st.columns(min(4, max(1, len(FEATURES))))
values = {}
for i, feat in enumerate(FEATURES):
    with cols[i % len(cols)]:
        values[feat] = st.number_input(feat, value=0.0)

def predict_one(model_key, Xrow_df):
    if model_key == "ANN" and "ANN" in models:
        Xp = fwd_X(Xrow_df, ann_cfg, ann_art["xsc"])
        yhat = models["ANN"].predict(Xp, verbose=0)
        return inv_Y(yhat, ann_cfg, ann_art["ysc"]).reshape(1, -1)
    if model_key == "RF" and "RF" in models:
        Xp = fwd_X(Xrow_df, rf_cfg, rf_art["xsc"])
        yhat = models["RF"].predict(Xp)
        return inv_Y(np.asarray(yhat).reshape(1,-1), rf_cfg, rf_art["ysc"])
    if model_key == "XGB" and "XGB" in models:
        Xp = fwd_X(Xrow_df, xgb_cfg, xgb_art["xsc"])
        yhat = models["XGB"].predict(Xp)
        return inv_Y(np.asarray(yhat).reshape(1,-1), xgb_cfg, xgb_art["ysc"])
    return None

if st.button("Predict"):
    if not FEATURES:
        st.error("Please provide feature names first.")
    else:
        Xrow = pd.DataFrame([values], columns=FEATURES)
        out = {}
        for key in ["ANN","RF","XGB"]:
            if key in models:
                pred = predict_one(key, Xrow)
                if pred is not None:
                    out[key] = pred[0]
        if out:
            st.success("Predictions")
            for k, arr in out.items():
                st.write(f"**{k}**")
                st.dataframe(pd.DataFrame(arr.reshape(1,-1), columns=TARGETS))
        else:
            st.warning("No predictions produced.")

st.markdown("---")
st.subheader("Batch prediction (CSV)")

up = st.file_uploader("Upload CSV with feature columns", type=["csv"])
if up is not None and FEATURES:
    try:
        df_in = pd.read_csv(up)
        st.write("Detected columns:", list(df_in.columns))
        dfX = df_in[FEATURES].copy()
        model_pick = st.selectbox("Choose model", [k for k in ["ANN","RF","XGB"] if k in models])
        if st.button("Run batch"):
            preds = predict_one(model_pick, dfX)
            preds_df = pd.DataFrame(preds, columns=TARGETS)
            st.dataframe(preds_df.head())
            fname = f"predictions_{model_pick}.csv"
            st.download_button("Download predictions", preds_df.to_csv(index=False), file_name=fname, mime="text/csv")
    except Exception as e:
        st.error(f"Failed to score: {e}")
