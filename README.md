# Wasserstein Strategic Classification (W2 WDRO)

A research-grade implementation of Wasserstein Distributionally Robust Optimization (W2 WDRO) for strategic classification under transport-bounded distribution shift.

This project studies robust learning against worst-case transported inputs using the Kantorovich dual formulation and adaptive dual optimization.

---

## Abstract

We investigate distributionally robust learning under transport-constrained strategic shifts. Given a base distribution P, we optimize classifier parameters to minimize worst-case risk over all distributions Q within a Wasserstein-2 ball of radius ρ around P. Using the Kantorovich dual representation, robust risk reduces to an inner loss-maximization problem regularized by a squared transport penalty and an outer minimization with adaptive dual updates. The framework supports feature immutability constraints, clean ERM baselines, and interactive exploration of robustness–accuracy trade-offs.

---

## Robust Learning Framework

We consider worst-case risk under a Wasserstein-2 constraint:

sup over Q such that W2(Q, P) ≤ ρ of E_Q[ loss_theta(x, y) ]

Using the Kantorovich dual, this becomes:

inf over λ ≥ 0 of

    λρ
    + E_{(x,y)~P} [
        sup over x' of (
            loss_theta(x', y)
            − λ ||x' − x||^2
        )
      ]

Where:

- ρ is the transport budget
- λ is the adaptive dual variable
- loss_theta(x, y) is the classification loss
- ||x' − x||^2 is the squared transport cost

The dual variable is updated as:

λ ← max(0, λ + η_lambda ( E[ ||x' − x||^2 ] − ρ ))

---

## Design Principles

This implementation emphasizes:

- Robust optimization under explicit transport constraints
- Adaptive dual variable learning
- Projected gradient-based inner maximization
- Immutable feature masking
- CPU-only reproducibility
- Fully interactive experimentation

---

## Experimental Capabilities

The framework allows you to:

- Compare ERM vs WDRO training
- Vary transport budget ρ
- Observe dual variable convergence
- Measure robust vs clean accuracy
- Visualize decision boundaries (2D case)

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

## Installation

pip install -r requirements.txt  
streamlit run app.py  

---

## Deployment

### Railway
Docker-based deployment using dynamic $PORT binding.

### Hugging Face Spaces
Compatible with CPU Docker Spaces.

---

## License

MIT License

---

## Citation

Haque, S. (2026). Wasserstein Strategic Classification (W2 WDRO). Interactive robust learning framework.






