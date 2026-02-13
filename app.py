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


st.set_page_config(page_title="W2 WDRO Lab", layout="wide")
st.title("Wasserstein Strategic Classification (W2 WDRO)")


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Data")
    n = st.slider("N samples", 300, 3000, 800, 100)
    d = st.slider("Feature dim", 2, 20, 8)
    seed = st.number_input("Seed", value=7)
    rho_corr = st.slider("Feature correlation", 0.0, 0.95, 0.3, 0.05)
    nonlinear = st.checkbox("Nonlinear boundary", value=True)
    label_noise = st.slider("Label noise", 0.0, 0.3, 0.05, 0.01)

    st.divider()
    st.header("Immutability")
    immutable_frac = st.slider("Immutable fraction", 0.0, 0.8, 0.2, 0.05)

    st.divider()
    st.header("Training")
    epochs = st.slider("Epochs", 20, 200, 80, 10)
    batch_size = st.slider("Batch size", 32, 512, 256, 32)
    lr = st.slider("Learning rate", 0.0005, 0.02, 0.005, 0.0005)

    st.divider()
    st.header("WDRO")
    rho = st.slider("Transport budget r", 0.01, 0.5, 0.08, 0.01)
    lam_init = st.slider("Initial lambda (kept >= 1)", 1.0, 20.0, 5.0)
    eta_lam = st.slider("Dual step", 0.01, 1.0, 0.3, 0.01)
    inner_steps = st.slider("Inner steps", 3, 15, 6)
    inner_step_size = st.slider("Inner step size", 0.05, 0.3, 0.12, 0.01)


# -------------------------
# Data
# -------------------------
X_np, y_np = make_synthetic(
    n=int(n),
    d=int(d),
    seed=int(seed),
    rho=float(rho_corr),
    nonlinear=bool(nonlinear),
    label_noise=float(label_noise),
)

rng = np.random.default_rng(int(seed))
mask = np.ones(d, dtype=np.float32)
num_immutable = int(immutable_frac * d)
if num_immutable > 0:
    imm_idx = rng.choice(d, num_immutable, replace=False)
    mask[imm_idx] = 0.0

X_t = torch.tensor(X_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
mask_t = torch.tensor(mask, dtype=torch.float32)

st.write(f"Dataset: N={n}, d={d}")


# -------------------------
# Train v4
# -------------------------
if st.button("Train W2-WDRO", type="primary"):

    model = TinyMLP(d, hidden=32)

    # Force lambda >= 1 so it never collapses to zero
    lam_init_safe = max(lam_init, 1.0)

    hist = train_wdro_w2_dual(
        model=model,
        X=X_t,
        y=y_t,
        mutable_mask=mask_t,
        rho=float(rho),
        lam_init=float(lam_init_safe),
        eta_lam=float(eta_lam),
        inner_steps=int(inner_steps),
        inner_step_size=float(inner_step_size),
        lr=float(lr),
        epochs=int(epochs),
        batch_size=int(batch_size),
        mix_clean_frac=0.0,
        seed=int(seed),
        max_wall_seconds=180,
        verbose_every=1,
    )

    if hist is None or len(hist) == 0:
        st.error("Training returned no epochs. Try lowering settings.")
    else:
        df = pd.DataFrame(hist)
        st.success(f"Training completed with {len(hist)} epochs.")

        st.dataframe(df.tail(10))

        fig1 = plt.figure()
        plt.plot(df["epoch"], df["loss_total"])
        plt.title("Objective")
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.plot(df["epoch"], df["avg_cost_sq"])
        plt.title("Transport cost")
        st.pyplot(fig2)

        fig3 = plt.figure()
        plt.plot(df["epoch"], df["lambda_dual"])
        plt.title("Lambda")
        st.pyplot(fig3)

        # Guarantee non-zero lambda display
        final_lambda = df["lambda_dual"].iloc[-1]
        if final_lambda == 0:
            st.warning("Lambda collapsed to 0. Increase inner_steps or inner_step_size.")
        else:
            st.metric("Final lambda", round(final_lambda, 4))
