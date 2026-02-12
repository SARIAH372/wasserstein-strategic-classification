
# Wasserstein Strategic Classification (W2 WDRO)

Research-grade implementation of W2 Wasserstein Distributionally Robust Optimization (WDRO) for strategic classification under transport-bounded distribution shift.

This framework trains a neural classifier against worst-case transported inputs using the Kantorovich dual formulation.

---

## Robust Objective (Kantorovich Dual Form)

We consider distributional robustness under a Wasserstein-2 transport constraint.

Robust risk can be written as:

sup_{Q: W_2(Q,P) ≤ ρ}  E_Q[ ℓ_θ(x, y) ]

Using the Kantorovich dual, this becomes:

inf_{λ ≥ 0} (
    λρ +
    E_{(x,y) ~ P} [
        sup_{x'} (
            ℓ_θ(x', y) − λ ||x' − x||²
        )
    ]
)

Where:

- ρ is the transport budget
- λ is the dual variable
- ℓ_θ is the classification loss
- ||x' − x||² is the W2 transport cost

---

## Feasible Set

Transported inputs satisfy:

- x' ∈ [0,1]^d
- x'_j = x_j for immutable features j

This enforces box constraints and feature immutability.

---

## Training Objective

The learner minimizes:

min_θ (
    λρ +
    E_{(x,y) ~ P} [
        sup_{x'} (
            ℓ_θ(x', y) − λ ||x' − x||²
        )
    ]
)

Dual update rule:

λ ← max(0, λ + η_λ ( E[||x' − x||²] − ρ ))

---

## Problem Setting

We study robust learning under:

- Strategic feature manipulation
- Transport-bounded distribution shift
- Worst-case loss maximization
- Adaptive dual optimization

---

## Key Components

- TinyMLP classifier (CPU-friendly)
- W2 Wasserstein inner adversary (projected gradient ascent)
- Immutable feature masking
- Adaptive dual variable update
- Clean ERM baseline
- Fully interactive Streamlit UI

---

## What You Can Explore

- Clean vs robust accuracy
- Effect of transport budget ρ
- Dual variable convergence
- Robustness under adversarial transport
- Decision boundary visualization (2D case)

---

## Project Structure

app.py  
requirements.txt  
Dockerfile  
README.md  
src/  
&nbsp;&nbsp;&nbsp;&nbsp;data.py  
&nbsp;&nbsp;&nbsp;&nbsp;model.py  
&nbsp;&nbsp;&nbsp;&nbsp;utils.py  
&nbsp;&nbsp;&nbsp;&nbsp;plots.py  
&nbsp;&nbsp;&nbsp;&nbsp;baseline/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;erm.py  
&nbsp;&nbsp;&nbsp;&nbsp;wdro/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;adversary.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;eval.py  

---

## Local Installation

pip install -r requirements.txt  
streamlit run app.py  

---

## Deployment

### Railway (recommended)
Uses Dockerfile with dynamic $PORT binding.

### Hugging Face Spaces
Docker-based deployment compatible with CPU instances.

---

## Notes

- Designed for CPU-only environments
- Inner maximization approximated via projected gradient ascent
- Synthetic correlated dataset generator included
- Intended for research and experimentation

---

## License

MIT License

---

