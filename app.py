# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data import make_synthetic
from src.model import TinyMLP
from src.baseline.erm import train_erm
from src.wdro.train import train_wdro_w2_dual
from src.wdro.eval import eval_wdro_metrics
from src.plots import plot_decision_2d, plot_curve


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Wasserstein Strategic Classification (W2 WDRO)",
    layout="wide",
)

st.title("Wasserstein Strategic Classification (W2 WDRO)")
st.caption(
    "Interactive research lab for W2 Wasserstein Distributionally Robust Optimization: "
    "inner loss-maximizing transport adversary + adaptive dual updates."
)


# -----------------------------
# Caching (important for hosted deployments)
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_data(n, d, seed, rho_corr, nonlinear, label_noise):
    X, y = make_synthetic(
        n=int(n),
        d=int(d),
        seed=int(seed),
        rho=float(rho_corr),
        nonlinear=bool(nonlinear),
        label_noise=float(label_noise),
    )
    return X, y

@st.cache_data(show_spinner=False)
def cached_mask(d, seed, immutable_frac):
    rng = np.random.default_rng(int(seed))
    d_int = int(d)
    m = int(np.floor((1.0 - float(immutable_frac)) * d_int))  # mutable count
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
    n = st.slider("N samples", 300, 6000, 1200, 100)
    d = st.slider("Feature dim d", 2, 40, 10, 1)
    seed = st.number_input("Seed", value=7, step=1)
    rho_corr = st.slider("Feature correlation ρ", 0.0, 0.95, 0.35, 0.05)
    nonlinear = st.checkbox("Nonlinear ground-truth", value=True)
    label_noise = st.slider("Label noise", 0.0, 0.30, 0.05, 0.01)

    st.divider()
    st.header("Immutability")
    immutable_frac = st.slider("Immutable feature fraction", 0.0, 0.95, 0.25, 0.05)

    st.divider()
    st.header("Model")
    hidden = st.slider("TinyMLP hidden size", 8, 128, 32, 8)
    lr = st.slider("Learning rate", 1e-4, 5e-2, 7e-3, 1e-4)
    epochs = st.slider("Epochs", 10, 800, 160, 10)
    batch_size = st.slider("Batch size", 16, 512, 256, 8)
    mix_clean = st.slider("Mix clean into robust batch (%)", 0, 100, 20, 5)

    st.divider()
    st.header("v4: W2 WDRO")
    rho_budget = st.slider("Transport budget r (target mean ||Δ||^2)", 0.000, 2.000, 0.080, 0.005)
    lam_init = st.slider("Initial lambda", 0.0, 50.0, 5.0, 0.5)
    eta_lam = st.slider("Dual step eta", 0.0, 5.0, 0.6, 0.05)
    inner_steps = st.slider("Inner adversary steps", 1, 50, 5, 1)
    inner_step_size = st.slider("Inner adversary step size", 0.001, 0.5, 0.06, 0.001)

    st.divider()
    st.header("Runtime safety (Railway-friendly)")
    max_wall_seconds = st.slider("Max training time (seconds)", 30, 900, 240, 30)
    verbose_every = st.slider("Log every N epochs", 1, 20, 1, 1)

    st.divider()
    show_2d = st.checkbox("Show 2D plots (only when d=2)", value=True)


# -----------------------------
# Data + tensors
# -----------------------------
X_np, y_np = cached_data(n, d, seed, rho_corr, nonlinear, label_noise)
mask_np, mutable_idx = cached_mask(d, seed, immutable_frac)

device = torch.device("cpu")
X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
y_t = torch.tensor(y_np, dtype=torch.float32, device=device).view(-1, 1)
mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)

st.write(
    f"**Dataset:** N={X_np.shape[0]}, d={X_np.shape[1]}  |  "
    f"**Mutable features:** {int(mask_np.sum())}/{int(d)}"
)

# -----------------------------
# Session state
# -----------------------------
if "model_erm" not in st.session_state:
    st.session_state.model_erm = None
if "model_v4" not in st.session_state:
    st.session_state.model_v4 = None
if "lambda_v4" not in st.session_state:
    st.session_state.lambda_v4 = float(lam_init)


# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["1) Theory", "2) Train ERM", "3) Train v4 (W2 WDRO)", "4) Compare"])

# ---- Tab 1: Theory ----
with tabs[0]:
    st.subheader("W2 WDRO (plain text)")
    st.write(
        "We optimize a classifier that remains reliable under worst-case distribution shift constrained by "
        "a Wasserstein-2 transport budget r.\n\n"
        "Conceptually:\n"
        "- Robust risk: maximize expected loss over all Q such that W2(Q, P) <= r.\n"
        "- Dual view: minimize over lambda >= 0 of lambda*r + E_{(x,y)~P}[ max_{x'} (loss_theta(x',y) - lambda*||x'-x||^2 ) ].\n\n"
        "In code:\n"
        "- Inner loop finds transported x' via projected gradient ascent.\n"
        "- Outer loop trains theta, and updates lambda to match the target transport budget."
    )

# ---- Tab 2: ERM ----
with tabs[1]:
    st.subheader("Train ERM baseline (clean)")
    colA, colB = st.columns([1.1, 0.9], gap="large")

    with colA:
        if st.button("Train ERM", type="primary"):
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
            st.success("ERM training complete.")

            df = pd.DataFrame(hist)
            st.dataframe(df.tail(12), use_container_width=True)
            st.pyplot(plot_curve(df["epoch"], df["loss"], "epoch", "loss", "ERM loss"))

    with colB:
        if st.session_state.model_erm is not None:
            m = eval_wdro_metrics(
                model=st.session_state.model_erm,
                X=X_t,
                y=y_t,
                mutable_mask=mask_t,
                lam_dual=float(st.session_state.lambda_v4),
                inner_steps=int(inner_steps),
                inner_step_size=float(inner_step_size),
            )
            st.metric("ERM clean acc", f"{m['acc_clean']:.3f}")
            st.metric("ERM robust acc (WDRO)", f"{m['acc_wdro_adv']:.3f}")
            st.metric("Mean transport cost (||Δ||^2)", f"{m['avg_cost_sq']:.4f}")

    if int(d) == 2 and show_2d and st.session_state.model_erm is not None:
        st.pyplot(plot_decision_2d(X_np, y_np, st.session_state.model_erm, title="ERM decision surface"))

# ---- Tab 3: v4 WDRO ----
with tabs[2]:
    st.subheader("Train v4: W2 Wasserstein DRO (dual + inner adversary)")
    st.write(
        "Tip for hosted CPU: start with small settings (epochs 60–120, inner steps 2–5, batch 256). "
        "The runtime safety cap ensures training always returns results instead of timing out."
    )

    if st.button("Train v4 (W2 WDRO)", type="primary"):
        model = TinyMLP(int(d), hidden=int(hidden))

        hist = train_wdro_w2_dual(
            model=model,
            X=X_t,
            y=y_t,
            mutable_mask=mask_t,
            rho=float(rho_budget),
            lam_init=float(lam_init),
            eta_lam=float(eta_lam),
            inner_steps=int(inner_steps),
            inner_step_size=float(inner_step_size),
            lr=float(lr),
            epochs=int(epochs),
            batch_size=int(batch_size),
            mix_clean_frac=float(mix_clean) / 100.0,
            seed=int(seed),
            max_wall_seconds=int(max_wall_seconds),   # <-- IMPORTANT (prevents 30+ min no-output)
            verbose_every=int(verbose_every),         # <-- logs progress to Railway logs
        )

        st.session_state.model_v4 = model
        st.session_state.lambda_v4 = float(hist[-1]["lambda_dual"]) if len(hist) else float(lam_init)

        st.success("v4 training finished (or stopped by time cap). Showing results below.")
        df = pd.DataFrame(hist)
        st.dataframe(df.tail(12), use_container_width=True)

        st.pyplot(plot_curve(df["epoch"], df["loss_total"], "epoch", "objective", "v4 objective"))
        st.pyplot(plot_curve(df["epoch"], df["avg_cost_sq"], "epoch", "mean ||Δ||^2", "transport cost"))
        st.pyplot(plot_curve(df["epoch"], df["lambda_dual"], "epoch", "lambda", "dual variable"))

        st.info(f"Final lambda stored for evaluation: {st.session_state.lambda_v4:.4f}")

# ---- Tab 4: Compare ----
with tabs[3]:
    st.subheader("Compare ERM vs v4 under the WDRO adversary")
    rows = []

    if st.session_state.model_erm is not None:
        m_erm = eval_wdro_metrics(
            model=st.session_state.model_erm,
            X=X_t,
            y=y_t,
            mutable_mask=mask_t,
            lam_dual=float(st.session_state.lambda_v4),
            inner_steps=int(inner_steps),
            inner_step_size=float(inner_step_size),
        )
        rows.append({"model": "ERM", **m_erm})

    if st.session_state.model_v4 is not None:
        m_v4 = eval_wdro_metrics(
            model=st.session_state.model_v4,
            X=X_t,
            y=y_t,
            mutable_mask=mask_t,
            lam_dual=float(st.session_state.lambda_v4),
            inner_steps=int(inner_steps),
            inner_step_size=float(inner_step_size),
        )
        rows.append({"model": "v4 W2 WDRO", **m_v4})

    if not rows:
        st.warning("Train ERM and/or v4 first.")
    else:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if int(d) == 2 and show_2d and st.session_state.model_v4 is not None:
        st.pyplot(plot_decision_2d(X_np, y_np, st.session_state.model_v4, title="v4 decision surface"))



   

       
   
        

        
        
