# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

 
from src.data import make_synthetic
from src.model import TinyMLP


from src.wdro.train import train_wdro_w2_dual
from src.wdro.eval import eval_wdro_metrics



# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Strategic Classification Lab v4 (W2 Wasserstein-DRO)",
    layout="wide",
)

st.title("Strategic Classification Lab v4 — W2 Wasserstein DRO (CPU, Hugging Face-ready)")
st.caption(
    "Research-grade robustness: train a classifier against worst-case transport-bounded distribution shift "
    "(W2 Wasserstein DRO via Kantorovich dual: inner loss-maximizing adversary + dual λ update)."
)

# -----------------------------
# Sidebar: controls
# -----------------------------
with st.sidebar:
    st.header("Data")
    n = st.slider("N samples", 300, 6000, 1500, 100)
    d = st.slider("Feature dim d", 2, 40, 10, 1)
    seed = st.number_input("Seed", value=7, step=1)
    rho_corr = st.slider("Feature correlation ρ", 0.0, 0.95, 0.35, 0.05)
    nonlinear = st.checkbox("Nonlinear ground-truth", value=True)
    label_noise = st.slider("Label noise", 0.0, 0.30, 0.05, 0.01)

    st.divider()
    st.header("Immutability / Action set")
    immutable_frac = st.slider("Immutable feature fraction", 0.0, 0.95, 0.25, 0.05)

    st.divider()
    st.header("Model (Leader)")
    hidden = st.slider("TinyMLP hidden size", 8, 128, 32, 8)
    lr = st.slider("Learning rate", 1e-4, 5e-2, 7e-3, 1e-4)
    epochs = st.slider("Epochs", 30, 800, 220, 10)
    batch = st.slider("Batch size", 16, 512, 128, 8)
    mix_clean = st.slider("Mix clean into robust batch (%)", 0, 100, 20, 5)

    st.divider()
    st.header("v4: W2 Wasserstein DRO")
    rho = st.slider("Transport budget ρ (mean ||Δ||² target)", 0.000, 2.000, 0.080, 0.005)
    lam_init = st.slider("Initial λ (dual)", 0.0, 50.0, 5.0, 0.5)
    eta_lam = st.slider("Dual step ηλ", 0.0, 5.0, 0.6, 0.05)
    wdro_steps = st.slider("Inner WDRO steps", 1, 50, 10, 1)
    wdro_step_size = st.slider("Inner WDRO step size", 0.001, 0.5, 0.06, 0.001)

    st.divider()
    show_2d = st.checkbox("Show 2D plots (only when d=2)", value=True)


# -----------------------------
# Data + masks
# -----------------------------
X_np, y_np = make_synthetic(
    n=int(n),
    d=int(d),
    seed=int(seed),
    rho=float(rho_corr),
    nonlinear=bool(nonlinear),
    label_noise=float(label_noise),
)

rng = np.random.default_rng(int(seed))
d_int = int(d)
m = int(np.floor((1.0 - float(immutable_frac)) * d_int))  # mutable count
perm = rng.permutation(d_int)
mutable_idx = np.sort(perm[:m])
mutable_mask_np = np.zeros(d_int, dtype=np.float32)
mutable_mask_np[mutable_idx] = 1.0

device = torch.device("cpu")
X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
y_t = torch.tensor(y_np, dtype=torch.float32, device=device).view(-1, 1)
mask_t = torch.tensor(mutable_mask_np, dtype=torch.float32, device=device)

st.write(
    f"**Dataset:** N={X_np.shape[0]}, d={X_np.shape[1]}  |  "
    f"**Mutable:** {int(mutable_mask_np.sum())}/{d_int}"
)

# Session state
if "model_erm" not in st.session_state:
    st.session_state.model_erm = None
if "model_v4" not in st.session_state:
    st.session_state.model_v4 = None
if "lambda_v4" not in st.session_state:
    st.session_state.lambda_v4 = float(lam_init)


# -----------------------------
# Helpers: plotting (2D)
# -----------------------------
def plot_decision_2d(X, y, model, title="Decision surface"):
    if X.shape[1] != 2:
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
    plt.scatter(X[:, 0], X[:, 1], s=12, c=y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    return fig


# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(
    [
        "1) Theory",
        "2) Train ERM",
        "3) Train v4 (W2-DRO)",
        "4) Compare (Clean vs WDRO)",
    ]
)

# ---- Tab 1: Theory ----
with tabs[0]:
    st.subheader("W2 Wasserstein DRO (Dual Form)")
    st.write(
        "We train the leader against worst-case distribution shift in a Wasserstein ball. "
        "Using the Kantorovich dual, robust risk becomes an inner maximization over transported points."
    )
    st.latex(
        r"\sup_{Q: W_2(Q,P)\le \rho}\; \mathbb{E}_Q[\ell_\theta(x,y)]"
        r"\;=\;\inf_{\lambda\ge0}\Big(\lambda\rho+\mathbb{E}_{(x,y)\sim P}\big[\sup_{x'} \ell_\theta(x',y)-\lambda\|x'-x\|_2^2\big]\Big)"
    )
    st.write(
        "**Implementation:** For each batch, we approximate the inner `sup` with projected gradient ascent on `x'` "
        "under box constraints `[0,1]^d` and an immutability mask. We update the dual variable `λ` to match the target transport budget `ρ`."
    )
    st.latex(r"\lambda \leftarrow \max\{0,\;\lambda+\eta_\lambda(\mathbb{E}[\|x'-x\|_2^2]-\rho)\}")

# ---- Tab 2: Train ERM ----
with tabs[1]:
    st.subheader("Train baseline ERM (clean training)")
    st.write("This gives a clean baseline before robust v4 training.")

    if st.button("Train ERM", type="primary"):
        model = TinyMLP(d_int, hidden=int(hidden))
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        bce = torch.nn.BCEWithLogitsLoss()

        hist = []
        # simple SGD minibatch loop
        idx = np.arange(X_np.shape[0])
        for ep in range(1, int(epochs) + 1):
            np.random.default_rng(int(seed) + ep).shuffle(idx)
            losses = []
            for i in range(0, len(idx), int(batch)):
                j = idx[i : i + int(batch)]
                xb = X_t[j]
                yb = y_t[j]

                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = bce(logits, yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())

            hist.append({"epoch": ep, "loss": float(np.mean(losses))})

        st.session_state.model_erm = model
        st.success("ERM training complete.")

        df = pd.DataFrame(hist)
        st.dataframe(df.tail(12), use_container_width=True)

        fig = plt.figure()
        plt.plot(df["epoch"], df["loss"])
        plt.xlabel("epoch")
        plt.ylabel("log loss")
        st.pyplot(fig)

        with torch.no_grad():
            p = torch.sigmoid(model(X_t))
            yhat = (p >= 0.5).float()
            acc = (yhat == y_t).float().mean().item()
        st.metric("Train accuracy (clean)", f"{acc:.3f}")

        if d_int == 2 and show_2d:
            st.pyplot(plot_decision_2d(X_np, y_np, model, "ERM decision surface"))

# ---- Tab 3: Train v4 ----
with tabs[2]:
    st.subheader("Train v4: W2 Wasserstein DRO (dual + inner loss-maximizing adversary)")
    st.write(
        "This is the upgrade: the inner adversary maximizes **classification loss** minus **λ·transport cost**, "
        "then the leader minimizes robust loss while λ adapts to enforce the transport budget ρ."
    )

    if st.button("Train v4 (W2-DRO)", type="primary"):
        model = TinyMLP(d_int, hidden=int(hidden))

        hist = train_wdro_w2_dual(
            model=model,
            X=X_t,
            y=y_t,
            mutable_mask=mask_t,
            rho=float(rho),
            lam_init=float(lam_init),
            eta_lam=float(eta_lam),
            inner_steps=int(wdro_steps),
            inner_step_size=float(wdro_step_size),
            lr=float(lr),
            epochs=int(epochs),
            batch_size=int(batch),
            mix_clean_frac=float(mix_clean) / 100.0,
            seed=int(seed),
        )

        st.session_state.model_v4 = model
        st.session_state.lambda_v4 = float(hist[-1]["lambda_dual"]) if len(hist) else float(lam_init)

        st.success("v4 W2-DRO training complete.")

        df = pd.DataFrame(hist)
        st.dataframe(df.tail(12), use_container_width=True)

        fig = plt.figure()
        plt.plot(df["epoch"], df["loss_total"])
        plt.xlabel("epoch")
        plt.ylabel("total objective")
        st.pyplot(fig)

        fig2 = plt.figure()
        plt.plot(df["epoch"], df["avg_cost_sq"])
        plt.axhline(float(rho), linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("avg transport cost (mean ||Δ||²)")
        st.pyplot(fig2)

        fig3 = plt.figure()
        plt.plot(df["epoch"], df["lambda_dual"])
        plt.xlabel("epoch")
        plt.ylabel("λ (dual)")
        st.pyplot(fig3)

        st.info(f"Stored final λ for evaluation: {st.session_state.lambda_v4:.4f}")

# ---- Tab 4: Compare ----
with tabs[3]:
    st.subheader("Compare ERM vs v4 under WDRO adversary")

    if st.session_state.model_erm is None and st.session_state.model_v4 is None:
        st.warning("Train at least one model (ERM or v4) first.")
    else:
        rows = []

        if st.session_state.model_erm is not None:
            m_erm = eval_wdro_metrics(
                model=st.session_state.model_erm,
                X=X_t,
                y=y_t,
                mutable_mask=mask_t,
                lam_dual=float(st.session_state.lambda_v4),  # evaluate both under same λ for fairness
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

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.write(
            "Notes:\n"
            "- `acc_clean`: accuracy on clean points\n"
            "- `acc_wdro_adv`: accuracy on worst-case transported points under the WDRO inner adversary\n"
            "- `avg_cost_sq`: mean transport cost (mean ||Δ||²), should hover near ρ after training\n"
        )

        if d_int == 2 and show_2d and st.session_state.model_v4 is not None:
            st.pyplot(plot_decision_2d(X_np, y_np, st.session_state.model_v4, "v4 W2-DRO decision surface (clean grid)"))


