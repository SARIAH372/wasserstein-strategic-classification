# app.py
import streamlit as st
import numpy as np
import pandas as pd
import torch

from src.data import make_synthetic
from src.model import TinyMLP
from src.baseline.erm import train_erm
from src.wdro.train import train_wdro_w2_dual
from src.wdro.eval import eval_wdro_metrics


# -----------------------------
# Basic page config
# -----------------------------
st.set_page_config(page_title="W2 WDRO Lab", layout="wide")
st.title("Strategic Classification Lab v4 — W2 Wasserstein DRO")
st.caption(
    "Train a classifier under transport-bounded worst-case shift (W2 WDRO): "
    "inner loss-maximizing adversary + adaptive dual updates."
)

# -----------------------------
# Simple plotting helpers (no extra modules required)
# -----------------------------
def plot_curve(x, y, xlabel: str, ylabel: str, title: str):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return fig

def plot_decision_2d(X_np, y_np, model, title="Decision surface"):
    import matplotlib.pyplot as plt
    import numpy as np
    if X_np.shape[1] != 2:
        fig = plt.figure()
        plt.text(0.05, 0.5, "2D plot available only when d=2")
        plt.axis("off")
        return fig

    x1 = np.linspace(0, 1, 250)
    x2 = np.linspace(0, 1, 250)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(grid))
        p = torch.sigmoid(logits).view(xx.shape).cpu().numpy()

    fig = plt.figure()
    plt.contourf(xx, yy, p, levels=25)
    plt.scatter(X_np[:, 0], X_np[:, 1], s=12, c=y_np)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    return fig

# -----------------------------
# Caching data + mask (important on Railway)
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
    epochs = st.slider("Epochs", 10, 800, 120, 10)
    batch_size = st.slider("Batch size", 16, 512, 256, 8)
    mix_clean = st.slider("Mix clean into robust batch (%)", 0, 100, 20, 5)

    st.divider()
    st.header("v4: W2 WDRO")
    rho_budget = st.slider("Transport budget r (target mean ||Δ||^2)", 0.000, 2.000, 0.080, 0.005)
    lam_init = st.slider("Initial lambda", 0.0, 50.0, 5.0, 0.5)
    eta_lam = st.slider("Dual step eta", 0.0, 5.0, 0.6, 0.05)
    wdro_steps = st.slider("Inner WDRO steps", 1, 50, 8, 1)
    wdro_step_size = st.slider("Inner WDRO step size", 0.001, 0.5, 0.08, 0.001)

    st.divider()
    st.header("Runtime safety (Railway)")
    max_wall_seconds = st.slider("Max training time (seconds)", 30, 900, 240, 30)
    verbose_every = st.slider("Log every N epochs", 1, 20, 1, 1)

    st.divider()
    show_2d = st.checkbox("Show 2D plots (only when d=2)", value=True)

# -----------------------------
# Data + tensors
# -----------------------------
X_np, y_np = cached_data(n, d, seed, rho_corr, nonlinear, label_noise)
mask_np, _ = cached_mask(d, seed, immutable_frac)

device = torch.device("cpu")
X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
y_t = torch.tensor(y_np, dtype=torch.float32, device=device).view(-1, 1)
mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)

st.write(f"**Dataset:** N={X_np.shape[0]}, d={X_np.shape[1]} | **Mutable:** {int(mask_np.sum())}/{int(d)}")

# -----------------------------
# Session state (CRUCIAL so results persist after reruns)
# -----------------------------
if "model_erm" not in st.session_state:
    st.session_state.model_erm = None
if "erm_hist" not in st.session_state:
    st.session_state.erm_hist = None

if "model_v4" not in st.session_state:
    st.session_state.model_v4 = None
if "lambda_v4" not in st.session_state:
    st.session_state.lambda_v4 = float(lam_init)
if "v4_hist" not in st.session_state:
    st.session_state.v4_hist = None

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Theory", "Train ERM", "Train v4 (W2-DRO)", "Compare"])

# ---- Theory
with tabs[0]:
    st.subheader("What this app does")
    st.write(
        "This lab explores W2 Wasserstein Distributionally Robust Optimization (WDRO) using a dual view:\n"
        "- Inner loop: finds transported inputs that increase loss while paying squared transport cost.\n"
        "- Outer loop: trains model parameters and updates the dual variable to match the transport budget.\n"
        "All experiments run on synthetic correlated data for reproducibility."
    )

# ---- Train ERM
with tabs[1]:
    st.subheader("Train baseline ERM (clean training)")
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

    # Always show ERM results if present
    if st.session_state.erm_hist:
        df = pd.DataFrame(st.session_state.erm_hist)
        st.dataframe(df.tail(20), use_container_width=True)
        st.pyplot(plot_curve(df["epoch"], df["loss"], "epoch", "loss", "ERM loss"))
        if int(d) == 2 and show_2d and st.session_state.model_erm is not None:
            st.pyplot(plot_decision_2d(X_np, y_np, st.session_state.model_erm, title="ERM decision surface"))
    else:
        st.caption("No ERM results yet. Click **Train ERM** to run.")

# ---- Train v4
with tabs[2]:
    st.subheader("Train v4: W2 WDRO (dual + inner adversary)")
    st.write("This is heavier than ERM. Use smaller settings for hosted CPU if needed.")

    if st.button("Train v4 (W2-DRO)", type="primary"):
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
                mix_clean_frac=float(mix_clean) / 100.0,
                seed=int(seed),
                max_wall_seconds=int(max_wall_seconds),   # ensures we ALWAYS get output
                verbose_every=int(verbose_every),
            )

        st.session_state.model_v4 = model
        st.session_state.v4_hist = hist
        st.session_state.lambda_v4 = float(hist[-1]["lambda_dual"]) if hist else float(lam_init)
        st.success("v4 W2-DRO training finished (or stopped by time cap).")

    # Always show v4 results if present
    if st.session_state.v4_hist:
        df = pd.DataFrame(st.session_state.v4_hist)
        st.dataframe(df.tail(20), use_container_width=True)

        st.pyplot(plot_curve(df["epoch"], df["loss_total"], "epoch", "objective", "v4 objective"))
        st.pyplot(plot_curve(df["epoch"], df["avg_cost_sq"], "epoch", "mean ||Δ||^2", "transport cost"))
        st.pyplot(plot_curve(df["epoch"], df["lambda_dual"], "epoch", "lambda", "dual variable"))

        st.info(f"Final lambda stored for evaluation: {st.session_state.lambda_v4:.4f}")

        # quick hint if adversary isn't moving points
        if float(df["avg_cost_sq"].iloc[-1]) == 0.0:
            st.warning(
                "avg_cost_sq is 0. Try increasing Inner WDRO steps/step size, or lowering Immutable fraction."
            )
    else:
        st.caption("No v4 results yet. Click **Train v4 (W2-DRO)** to run.")

# ---- Compare
with tabs[3]:
    st.subheader("Compare ERM vs v4 under WDRO adversary")

    rows = []
    if st.session_state.model_erm is not None:
        m_erm = eval_wdro_metrics(
            model=st.session_state.model_erm,
            X=X_t,
            y=y_t,
            mutable_mask=mask_t,
            lam_dual=float(st.session_state.lambda_v4),
            inner_steps=int(wdro_steps),
            inner_step_size=float(wdro_step_size),
        )
        rows.append({"model": "ERM", **m_erm})

    if st.session_state.model_v4 is not None:
        m_v4 = eval_wdro_metrics(
            model=st.session_state.model_v4,
            X=X_t,
            y=y_t,
            mutable_mask=mask_t,
            lam_dual=float(st.session_state.lambda_v4),
            inner_steps=int(wdro_steps),
            inner_step_size=float(wdro_step_size),
        )
        rows.append({"model": "v4 W2-DRO", **m_v4})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("Train ERM and/or v4 first, then return here to compare metrics.")


   
    


      
            
