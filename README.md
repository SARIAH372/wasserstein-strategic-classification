# Wasserstein Strategic Classification (W2 WDRO)

Research-grade implementation of W2 Wasserstein Distributionally Robust Optimization (WDRO) for strategic classification under transport-bounded distribution shift.

This framework trains a neural classifier against worst-case transported inputs using the Kantorovich dual formulation.

---

## Robust Objective (Intuitive Form)

We study worst-case risk under a Wasserstein-2 transport constraint.

The robust objective can be written as:

> sup over Q such that W₂(Q,P) ≤ ρ of E_Q[ℓ_θ(x, y)]

Using the Kantorovich dual representation, this becomes:

> inf over λ ≥ 0 of  
> λρ + E_{(x,y) ~ P} [ sup over x' of ( ℓ_θ(x', y) − λ‖x' − x‖² ) ]

Where:

- ρ is the transport budget  
- λ is the dual variable  
- ℓ_θ(x,y) is the classification loss  
- ‖x' − x‖² is the squared transport cost  

---

## Feasible Set

Transported inputs satisfy:

- x' ∈ [0,1]^d  
- x'_j = x_j for immutable features j  

This enforces both box constraints and feature immutability.

---

## Training Objective

The learner minimizes:

> λρ + E_{(x,y) ~ P} [ sup over x' of ( ℓ_θ(x', y) − λ‖x' − x‖² ) ]

The dual variable is updated adaptively:

> λ ← max(0, λ + η_λ ( E[‖x' − x‖²] − ρ ))

This ensures the average transport cost stays near the target budget ρ.

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
- W2 Wasserstein inner adversary via projected gradient ascent
- Immutable feature masking
- Adaptive dual variable update
- Clean ERM baseline
- Fully interactive Streamlit UI

---

## What You Can Explore

- Clean vs robust accuracy
- Effect of transport budget ρ
- Dual variable convergence behavior
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

## Citation 

Sariah Haque (2026). Wasserstein Strategic Classification (W2 WDRO). Interactive robust learning framework.



