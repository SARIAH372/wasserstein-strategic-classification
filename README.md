# Wasserstein Strategic Classification (W2 WDRO)

Research-grade implementation of **W2 Wasserstein Distributionally Robust Optimization (WDRO)** for strategic classification under transport-bounded distribution shift.

This framework trains a neural classifier against worst-case transported inputs using the Kantorovich dual formulation.

---

## 🔬 Robust Objective (Kantorovich Dual Form)

```math
\begin{aligned}
\sup_{Q:\, W_2(Q,P)\le \rho}\; \mathbb{E}_Q[\ell_\theta(x,y)]
&=
\inf_{\lambda\ge 0}\Bigg(
\lambda\rho
+
\mathbb{E}_{(x,y)\sim P}\Big[
\sup_{x'\in \mathcal{X}}
\Big(
\ell_\theta(x',y)
-\lambda\|x'-x\|_2^2
\Big)
\Big]
\Bigg).
\end{aligned}
Where the feasible set enforces transport and immutability constraints:

𝑋
=
{
𝑥
′
∈
[
0
,
1
]
𝑑
:
  
𝑥
𝑗
′
=
𝑥
𝑗
 for immutable features 
𝑗
}
.
X={x
′
∈[0,1]
d
:x
j
′
	​

=x
j
	​

 for immutable features j}.
🧠 Problem Setting

We study robust learning under:

Strategic feature manipulation

Transport-bounded distribution shift

Worst-case loss maximization

Adaptive dual optimization

The learner minimizes robust risk:

min
⁡
𝜃
  
(
𝜆
𝜌
+
𝐸
(
𝑥
,
𝑦
)
∼
𝑃
[
sup
⁡
𝑥
′
∈
𝑋
(
ℓ
𝜃
(
𝑥
′
,
𝑦
)
−
𝜆
∥
𝑥
′
−
𝑥
∥
2
2
)
]
)
.
θ
min
	​

(λρ+E
(x,y)∼P
	​

[
x
′
∈X
sup
	​

(ℓ
θ
	​

(x
′
,y)−λ∥x
′
−x∥
2
2
	​

)]).
	​


Dual update rule:

𝜆
←
max
⁡
{
0
,
  
𝜆
+
𝜂
𝜆
(
𝐸
[
∥
𝑥
′
−
𝑥
∥
2
2
]
−
𝜌
)
}
.
λ←max{0,λ+η
λ
	​

(E[∥x
′
−x∥
2
2
	​

]−ρ)}.
⚙️ Key Components

TinyMLP classifier (CPU friendly)

W2 Wasserstein inner adversary (projected gradient ascent)

Immutable feature masking

Adaptive dual variable update

Clean ERM baseline

Fully interactive Streamlit UI

📊 What You Can Explore

Clean vs robust accuracy

Effect of transport budget 
𝜌
ρ

Dual variable convergence behavior

Robustness under adversarial transport

Decision boundary visualization (2D)
📦 Project Structure
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
🚀 Deployment
Railway (recommended)

Uses Dockerfile with dynamic $PORT binding.

Hugging Face Spaces

Docker-based deployment compatible with CPU instances.
🖥️ Local Installation
pip install -r requirements.txt
streamlit run app.py
🧪 Example Experiments

Sweep 
𝜌
ρ to trace a robustness frontier

Compare ERM vs WDRO adversarial accuracy

Study λ convergence dynamics

Vary inner adversary steps and step size

🧾 License

MIT License
