# Wasserstein Strategic Classification (W2 WDRO)

Research-grade implementation of Wasserstein Distributionally Robust Optimization (W2 WDRO) for strategic classification under transport-bounded distribution shift.

This project trains a neural classifier against worst-case transported inputs using a dual robust-optimization view: an inner loss-maximizing adversary plus adaptive dual updates.

---

## Overview

**Goal:** Train a classifier that remains reliable not only on the observed distribution `P`, but also under worst-case distribution shift constrained by a Wasserstein-2 transport budget.

**Core idea:** Robust risk can be written as an outer optimization that depends on an inner “adversary” which searches for transported inputs that increase loss, while paying a squared transport cost.

---

## Robust Objective (plain text)

Robust risk (conceptually):

- Maximize expected loss over all distributions `Q` such that `W2(Q, P) <= r`.

Dual form (conceptually):

- Minimize over `lambda >= 0` the quantity:

  - `lambda * r`
  - `+ E_{(x,y)~P} [ max over x_prime of ( loss_theta(x_prime, y) - lambda * ||x_prime - x||^2 ) ]`

Definitions:

- `r` = transport budget (W2 radius)
- `lambda` = dual variable
- `loss_theta(x, y)` = classification loss for parameters `theta`
- `||x_prime - x||^2` = squared L2 transport cost

Dual update rule (conceptually):

- `lambda <- max(0, lambda + eta * (avg_cost - r))`

---

## Constraints used in this repo

Transported inputs `x_prime` satisfy:

- Box constraints: each feature is in `[0, 1]`
- Immutability: selected features are fixed (`x_prime[j] = x[j]` for immutable indices)

---

## What this repo implements

- ERM baseline (clean training)
- W2 WDRO training (dual + inner adversary)
- Inner adversary via projected gradient ascent
- Adaptive dual updates to match the target transport budget
- Interactive Streamlit interface for experimentation
- Synthetic correlated dataset generator for reproducible testing

---

## What you can explore

- Clean accuracy vs robust (adversarial-transport) accuracy
- Effect of transport budget `r` on robustness
- Dual variable behavior during training
- Sensitivity to inner adversary settings (steps, step size)
- Decision boundary visualization (2D case)

---

## Repository structure

- `app.py`
- `requirements.txt`
- `Dockerfile`
- `README.md`
- `src/`
  - `data.py`
  - `model.py`
  - `utils.py`
  - `plots.py`
  - `baseline/`
    - `erm.py`
  - `wdro/`
    - `adversary.py`
    - `train.py`
    - `eval.py`

---

## Local run

1) Install dependencies: `pip install -r requirements.txt`  
2) Start the app: `streamlit run app.py`

---

## Deployment

- Railway: Docker-based deployment using dynamic `PORT` binding.
- Hugging Face Spaces: Docker-based deployment compatible with CPU instances.

---

## License

MIT License

---

## Citation 

Sariah Haque (2026). Wasserstein Strategic Classification (W2 WDRO). Interactive robust learning framework.


