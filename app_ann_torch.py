import os
import copy
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import torch
import torch.nn as nn

# -----------------------------
# 1) Load artifacts
# -----------------------------
ART_DIR = Path(".")  # current folder; change if needed

meta   = joblib.load(ART_DIR / "meta.joblib")
Xsc    = joblib.load(ART_DIR / "Xsc.pkl")
Ysc    = joblib.load(ART_DIR / "Ysc.pkl")
params = joblib.load(ART_DIR / "best_params.joblib")      # dict of best hyperparams
fold_states = joblib.load(ART_DIR / "fold_states.joblib") # list of state_dicts (one per fold)

FEATURES = meta.get("X_columns") or meta.get("features") or meta.get("X_cols")
YVARS    = meta.get("Y_columns") or meta.get("targets")  or meta.get("Y_cols")
log_X    = bool(meta.get("log_transform_X", False) or meta.get("log_x", False))
log_Y    = bool(meta.get("log_transform_Y", False) or meta.get("log_y", False))

if FEATURES is None or YVARS is None:
    raise RuntimeError("meta.joblib must contain feature/target names (X_columns / Y_columns).")

DEVICE = "cpu"  # inference on CPU is fine

# -----------------------------
# 2) Rebuild your ANN exactly
# -----------------------------
ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}

class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden, act="relu", drop=0.15):
        super().__init__()
        A = ACTS[act]
        L = []
        p = in_d
        for h in hidden:
            L += [nn.Linear(p, h), A()]
            L += [nn.Dropout(drop)]
            p = h
        L += [nn.Linear(p, out_d)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        return self.net(x)

class MultiHead(nn.Module):
    def __init__(self, in_d, out_d, trunk, head, act="relu", drop=0.15):
        super().__init__()
        A = ACTS[act]
        T = []
        p = in_d
        for h in trunk:
            T += [nn.Linear(p, h), A()]
            T += [nn.Dropout(drop)]
            p = h
        self.trunk = nn.Sequential(*T)
        self.heads = nn.ModuleList()
        for _ in range(out_d):
            H = []
            ph = p
            for h in head:
                H += [nn.Linear(ph, h), A()]
                ph = h
            H += [nn.Linear(ph, 1)]
            self.heads.append(nn.Sequential(*H))
    def forward(self, x):
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=1)

def build_model(n_in, n_out, cfgp, use_multihead=True, head_sizes=None):
    hidden_sizes = cfgp.get("hidden_sizes", [50, 25, 10])
    activation   = cfgp.get("activation", "elu")
    dropout      = cfgp.get("dropout", 0.30)
    if use_multihead and n_out > 1:
        head_sizes = head_sizes or [64]
        model = MultiHead(n_in, n_out, hidden_sizes, head_sizes, activation, dropout)
    else:
        model = MLP(n_in, n_out, hidden_sizes, activation, dropout)
    return model.to(DEVICE)

def predict_with_states(X_np, fold_states, params, n_out):
    """Average predictions from all fold state_dicts."""
    X_t = torch.from_numpy(X_np.astype(np.float32)).to(DEVICE)
    preds = []
    for sd in fold_states:
        mdl = build_model(
            n_in=X_np.shape[1], n_out=n_out, cfgp=params,
            use_multihead=params.get("use_multihead", True),
            head_sizes=params.get("head_sizes", [64])
        )
        mdl.load_state_dict(sd)
        mdl.eval()
        with torch.no_grad():
            yp = mdl(X_t).cpu().numpy()
        preds.append(yp)
    return np.mean(preds, axis=0)

# -----------------------------
# 3) Pre / post-processing
# -----------------------------
def fwd_X(df_or_np):
    X = df_or_np.values if hasattr(df_or_np, "values") else np.asarray(df_or_np)
    X = X.astype(np.float32, copy=False)
    if log_X:
        # safe log1p-like (use your exact shift if you used one in training)
        eps = 1e-9
        X = np.log(X + eps)
    Xz = Xsc.transform(X)
    return Xz

def inv_Y(Yz):
    Y = Ysc.inverse_transform(Yz)
    if log_Y:
        Y = np.exp(Y)  # undo log
    return Y

# -----------------------------
# 4) Streamlit UI
# -----------------------------
st.set_page_config(page_title="ANN Predictor (PyTorch)", layout="wide")
st.title("ANN Predictor — PyTorch (Ensemble of CV folds)")

st.sidebar.header("Model info")
st.sidebar.write(f"Inputs: {len(FEATURES)}")
st.sidebar.write(f"Outputs: {len(YVARS)}")
st.sidebar.write(f"Log X: {log_X} | Log Y: {log_Y}")
st.sidebar.write(f"Uses multihead: {bool(params.get('use_multihead', True))}")
st.sidebar.write(f"Hidden: {params.get('hidden_sizes', [50,25,10])}")
st.sidebar.write(f"Head sizes: {params.get('head_sizes', [64])}")

# --- Single-row prediction
st.subheader("Single prediction")
cols = st.columns(3)
inputs = []
for i, name in enumerate(FEATURES):
    with cols[i % 3]:
        val = st.number_input(name, value=0.0, format="%.6f")
        inputs.append(val)

if st.button("Predict (single)"):
    X_in = np.array(inputs, dtype=np.float32).reshape(1, -1)
    Xz   = fwd_X(X_in)
    Yz   = predict_with_states(Xz, fold_states, params, n_out=len(YVARS))
    Yo   = inv_Y(Yz)
    df   = pd.DataFrame(Yo, columns=YVARS)
    st.success("Prediction")
    st.dataframe(df.style.format("{:.6f}"))
    st.download_button("Download CSV", df.to_csv(index=False), "ann_pred_single.csv", "text/csv")

st.markdown("---")

# --- Batch CSV prediction
st.subheader("Batch prediction (CSV)")
up = st.file_uploader("Upload CSV with EXACT columns (same order) as training features", type=["csv"])
if up is not None:
    try:
        df_in = pd.read_csv(up)
        if list(df_in.columns) != list(FEATURES):
            st.error("CSV columns must match the training feature names and order.")
        else:
            Xz = fwd_X(df_in)
            Yz = predict_with_states(Xz, fold_states, params, n_out=len(YVARS))
            Yo = inv_Y(Yz)
            out = pd.DataFrame(Yo, columns=YVARS)
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head().style.format("{:.6f}"))
            st.download_button("Download predictions", out.to_csv(index=False),
                               file_name="ann_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to score file: {e}")
