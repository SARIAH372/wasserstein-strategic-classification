


import json
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data import make_synthetic
from src.model import TinyMLP
from src.baseline.erm import train_erm
from src.wdro.train import train_wdro_w2_dual
from src.wdro.eval import eval_wdro_metrics


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="W2 WDRO Lab", layout="wide")
st.title("Wasserstein Strategic Classification (W2 WDRO)")
st.caption("Empirical study of W2 Wasserstein Distributionally Robust Optimization under transport-constrained shift.")


# -----------------------------
# Plot helper
# -----------------------------
def plot_curve(x, y, xlabel: str, ylabel: str, title: str):
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return fig


# -----------------------------
# Cache data + mask
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_data(n, d, seed, rho_corr, nonlinear, label_noise):
    return make_synthetic(
        n=int(n),
        d=int(d),
        seed=int(seed),
        rho=float(rho_corr),
        nonlinear=bool(nonlinear),
        label_noise=float(label_noise),
    )

@st.cache_data(show_spinner=False)
def cached_mask(d, seed, immutable_frac):
    rng = np.random.default_rng(int(seed))
    d_int = int(d)
    m = int(np.floor((1.0 - float(immutable_frac)) * d_int))
    perm = rng.permutation(d_int)
    mutable_idx = np.sort(perm[:m])
    mask = np.zeros(d_int, dtype=np.float32)
    mask[mutable_idx] = 1.0
    return mask, mutable_idx


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Data")
    n = st.slider("N samples", 300, 6000, 1000, 100)
    d = st.slider("Feature dim d", 2, 40, 8, 1)
    seed = st.number_input("Seed", value=7, step=1)
    rho_corr = st.slider("Feature correlation", 0.0, 0.95, 0.35, 0.05)
    nonlinear = st.checkbox("Nonlinear boundary", value=True)
    label_noise = st.slider("Label noise", 0.0, 0.30, 0.05, 0.01)

    st.divider()
    st.header("Immutability")
    immutable_frac = st.slider("Immutable feature fraction", 0.0, 0.95, 0.25, 0.05)

    st.divider()
    st.header("Training")
    hidden = st.slider("TinyMLP hidden size", 8, 128, 32, 8)
    lr = st.slider("Learning rate", 1e-4, 5e-2, 7e-3, 1e-4)
    epochs = st.slider("Epochs", 10, 800, 120, 10)
    batch_size = st.slider("Batch size", 16, 512, 256, 8)

    st.divider()
    st.header("v4: W2 WDRO")
    rho_budget = st.slider("Transport budget r", 0.000, 2.000, 0.080, 0.005)
    lam_init = st.slider("Initial lambda", 0.0, 50.0, 5.0, 0.5)
    eta_lam = st.slider("Dual step eta", 0.0, 5.0, 0.6, 0.05)
    wdro_steps = st.slider("Inner WDRO steps", 1, 50, 8, 1)
    wdro_step_size = st.slider("Inner WDRO step size", 0.001, 0.5, 0.08, 0.001)

    st.divider()
    max_wall_seconds = st.slider("Max training time (seconds)", 30, 900, 240, 30)


# -----------------------------
# Data preparation
# -----------------------------
X_np, y_np = cached_data(n, d, seed, rho_corr, nonlinear, label_noise)
mask_np, _ = cached_mask(d, seed, immutable_frac)

X_t = torch.tensor(X_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
mask_t = torch.tensor(mask_np, dtype=torch.float32)

st.write(f"Dataset: N={X_np.shape[0]}, d={X_np.shape[1]} | Mutable: {int(mask_np.sum())}/{int(d)}")


# -----------------------------
# Session state
# -----------------------------
if "model_erm" not in st.session_state:
    st.session_state.model_erm = None
if "erm_hist" not in st.session_state:
    st.session_state.erm_hist = None

if "model_v4" not in st.session_state:
    st.session_state.model_v4 = None
if "v4_hist" not in st.session_state:
    st.session_state.v4_hist = None
if "lambda_v4" not in st.session_state:
    st.session_state.lambda_v4 = float(lam_init)


# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Theory", "Train ERM", "Train v4 (W2 WDRO)", "Compare"])


# -----------------------------
# Theory tab
# -----------------------------
with tabs[0]:
    st.subheader("W2 Wasserstein Distributionally Robust Optimization")

    st.markdown(
        """
This application studies classification under transport-constrained distribution shift using
Wasserstein-2 Distributionally Robust Optimization (WDRO).

Given a reference distribution P, WDRO optimizes against worst-case shifted distributions Q
within a transport budget r.

Training alternates between:
- an inner adversarial transport step that increases loss while penalizing large movement, and
- an outer model update that improves robustness under this constrained shift.
"""
    )


# -----------------------------
# Train ERM
# -----------------------------
with tabs[1]:
    st.subheader("Train ERM baseline")

    if st.button("Train ERM", type="primary"):
        with st.spinner("Training ERM..."):
            model = TinyMLP(int(d), hidden=int(hidden))
            hist = train_erm(
                model=model,
                X=X_t,
                y=y_t,
                lr=float(lr),
                epochs=int(epochs),
                batch_size=int(batch_size),
                seed=int(seed),
            )
        st.session_state.model_erm = model
        st.session_state.erm_hist = hist
        st.success("ERM training complete.")

    if st.session_state.erm_hist is not None:
        df = pd.DataFrame(st.session_state.erm_hist)
        st.dataframe(df.tail(20), use_container_width=True)
        st.pyplot(plot_curve(df["epoch"], df["loss"], "epoch", "loss", "ERM loss"))
    else:
        st.caption("No ERM results yet.")


# -----------------------------
# Train v4 WDRO (always show outcome; show exceptions)
# -----------------------------
with tabs[2]:
    st.subheader("Train v4: W2 WDRO")

    if st.button("Train v4 (W2 WDRO)", type="primary"):
        try:
            with st.spinner("Training v4 W2-WDRO..."):
                model = TinyMLP(int(d), hidden=int(hidden))

                hist = train_wdro_w2_dual(
                    model=model,
                    X=X_t,
                    y=y_t,
                    mutable_mask=mask_t,
                    rho=float(rho_budget),
                    lam_init=float(lam_init),
                    eta_lam=float(eta_lam),
                    inner_steps=int(wdro_steps),
                    inner_step_size=float(wdro_step_size),
                    lr=float(lr),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    mix_clean_frac=0.0,
                    seed=int(seed),
                    max_wall_seconds=int(max_wall_seconds),
                    verbose_every=1,
                )

            st.session_state.model_v4 = model
            st.session_state.v4_hist = hist  # store as-is (can be [])
            if hist and len(hist) > 0:
                st.session_state.lambda_v4 = float(hist[-1]["lambda_dual"])

            st.success(f"v4 training finished. epochs_returned={0 if hist is None else len(hist)}")

        except Exception as e:
            st.session_state.v4_hist = None
            st.error("WDRO training crashed. Full error below:")
            st.exception(e)

    # SHOW RESULTS EVEN IF EMPTY LIST
    if st.session_state.v4_hist is not None:
        st.write("epochs_returned =", len(st.session_state.v4_hist))

        df = pd.DataFrame(st.session_state.v4_hist)
        if len(df) == 0:
            st.warning("Training returned 0 epochs. Increase max training time or reduce settings.")
        else:
            st.dataframe(df.tail(20), use_container_width=True)
            st.pyplot(plot_curve(df["epoch"], df["loss_total"], "epoch", "objective", "v4 objective"))
            st.pyplot(plot_curve(df["epoch"], df["avg_cost_sq"], "epoch", "mean ||Δ||^2", "transport cost"))
            st.pyplot(plot_curve(df["epoch"], df["lambda_dual"], "epoch", "lambda", "dual variable"))

            # show cost precisely
            st.write("avg_cost_sq (last) =", f"{float(df['avg_cost_sq'].iloc[-1]):.10e}")

            st.metric("Final lambda", f"{st.session_state.lambda_v4:.4f}")
    else:
        st.caption("No v4 results yet.")


# -----------------------------
# Compare (adds ONLY a toggle; training unchanged)
# -----------------------------
with tabs[3]:
    st.subheader("Compare ERM vs v4")

    # Evaluation-only stress test: makes differences visible even when margins are large.
    stress_test = st.checkbox("Stress-test evaluation (stronger adversary)", value=True)

    if stress_test:
        lam_eval = 0.0
        eval_steps = max(int(wdro_steps), 30)
        eval_step = max(float(wdro_step_size), 0.20)
    else:
        lam_eval = float(st.session_state.lambda_v4)
        eval_steps = int(wdro_steps)
        eval_step = float(wdro_step_size)

    rows = []

    if st.session_state.model_erm is not None:
        m_erm = eval_wdro_metrics(
            model=st.session_state.model_erm,
            X=X_t,
            y=y_t,
            mutable_mask=mask_t,
            lam_dual=lam_eval,
            inner_steps=eval_steps,
            inner_step_size=eval_step,
        )
        rows.append({"model": "ERM", **m_erm})

    if st.session_state.model_v4 is not None:
        m_v4 = eval_wdro_metrics(
            model=st.session_state.model_v4,
            X=X_t,
            y=y_t,
            mutable_mask=mask_t,
            lam_dual=lam_eval,
            inner_steps=eval_steps,
            inner_step_size=eval_step,
        )
        rows.append({"model": "v4 W2-WDRO", **m_v4})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("Train models first.")
