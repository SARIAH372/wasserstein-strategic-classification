# Wasserstein Strategic Classification (W2 WDRO)

Research-grade implementation of **W2 Wasserstein Distributionally Robust Optimization (WDRO)** for strategic classification under transport-bounded distribution shift.

This framework trains a neural classifier against worst-case transported inputs using the Kantorovich dual formulation.

---

## 🔬 Robust Objective (Kantorovich Dual Form)

```math
\sup_{Q: W_2(Q,P)\le \rho}\mathbb{E}_Q[\ell_\theta(x,y)]
=
\inf_{\lambda\ge0}
\left(
\lambda\rho+
\mathbb{E}_{(x,y)\sim P}
\left[
\sup_{x'} \ell_\theta(x',y)
- \lambda \|x'-x\|_2^2
\right]
\right)
The inner supremum is approximated via projected gradient ascent under:

Box constraints: 
𝑥
′
∈
[
0
,
1
]
𝑑
x
′
∈[0,1]
d

Immutable feature masks

Transport penalty 
∥
𝑥
′
−
𝑥
∥
2
2
∥x
′
−x∥
2
2
	​


The dual variable 
𝜆
λ is updated to enforce the target transport budget 
𝜌
ρ.
