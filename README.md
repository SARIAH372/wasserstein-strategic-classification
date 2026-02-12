# Wasserstein Strategic Classification (W2 WDRO)

Research-grade implementation of W2 Wasserstein Distributionally Robust Optimization (WDRO) for strategic classification under transport-bounded distribution shift.

This framework trains a neural classifier against worst-case transported inputs using the Kantorovich dual formulation:

\[
\sup_{Q: W_2(Q,P)\le \rho}\mathbb{E}_Q[\ell_\theta(x,y)]
=
\inf_{\lambda\ge0}\left(
\lambda\rho+
\mathbb{E}_{(x,y)\sim P}
\left[
\sup_{x'} \ell_\theta(x',y) - \lambda \|x'-x\|_2^2
\right]
\right)
\]

The inner supremum is approximated via projected gradient ascent under box constraints and immutable feature masks. The dual variable λ is updated to enforce a target transport budget ρ.

---

## 🔬 Problem Setting

We study robust learning under:

- **Strategic feature manipulation**
- **Transport-bounded distribution shift**
- **Worst-case loss maximization**
- **Adaptive dual optimization**

The learner minimizes robust risk:

\[
\min_\theta \lambda\rho + 
\mathbb{E}_{P}
\left[
\sup_{x'} \ell_\theta(x',y) - \lambda\|x'-x\|_2^2
\right]
\]

with dual update:

\[
\lambda \leftarrow \max(0,\lambda + \eta_\lambda(\mathbb{E}[\|x'-x\|_2^2]-\rho))
\]

---

## 🧠 Key Components

- TinyMLP classifier (CPU friendly)
- W2 Wasserstein inner adversary
- Immutable feature masking
- Adaptive dual variable update
- Clean ERM baseline for comparison
- Fully interactive Streamlit UI

---

## 📊 What You Can Explore

- Clean vs robust accuracy
- Effect of transport budget ρ
- Dual variable dynamics
- Robustness under adversarial transport
- Decision boundary visualization (2D)

---

## 📦 Project Structure

app.py
requirements.txt
Dockerfile
README.md
src/
data.py
model.py
utils.py
plots.py
baseline/
erm.py
wdro/
adversary.py
train.py
eval.py

---

## 🚀 Deployment

### Railway (recommended)
Uses Dockerfile with dynamic `$PORT` binding.

### Hugging Face Spaces
Docker-based deployment compatible with CPU instances.

---

## ⚙️ Installation (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
🧪 Example Experiments

Vary ρ to trace robustness frontier

Compare ERM vs WDRO adversarial accuracy

Observe λ convergence behavior

Study stability under different inner step sizes

🧾 License

MIT License
