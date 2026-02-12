# Wasserstein Strategic Classification (W2 WDRO)

Research-grade implementation of Wasserstein Distributionally Robust Optimization (W2 WDRO) for strategic classification under transport-bounded distribution shift.

This project trains a neural classifier against worst-case transported inputs using the Kantorovich dual view (inner loss-maximizing adversary + adaptive dual updates).

---

## Summary

Goal:
Train a classifier that performs well not only on the observed data distribution P, but also under worst-case distribution shifts constrained by a Wasserstein-2 transport budget.

Key idea:
Worst-case distribution shift can be handled via a dual formulation that turns robustness into (1) an inner adversary that searches for hard transported inputs and (2) an outer learner update that improves robustness.

---

## Robust Objective (Plain Text)

Robust risk:
Maximize expected loss over all distributions Q that are within a W2 distance budget r from the base distribution P.

Dual form (plain text):
Minimize over dual variable lambda >= 0 of:

lambda * r
+ E over (x,y) from P of [
    maximize over x_prime of (
        loss_theta(x_prime, y) - lambda * squared_distance(x_prime, x)
    )
  ]

Where:
- r is the transport budget (W2 radius)
- lambda is the dual variable
- loss_theta is the classification loss under model parameters theta
- squared_distance(x_prime, x) is the squared L2 transport cost

Dual update (plain text):
lambda <- max(0, lambda + eta * ( average_squared_distance - r ))

---

## Constraints Used in This Repo

Transported inputs x_prime satisfy:
- Box constraints: each feature lies in [0, 1]
- Immutability: selected features are fixed and cannot change

---

## What This Repo Implements

- ERM baseline (clean training)
- W2 WDRO training (dual + inner adversary)
- Inner adversary via projected gradient ascent
- Adaptive dual variable updates to match the target transport budget
- Interactive Streamlit interface for experimentation
- Synthetic correlated dataset generator for reproducible tests

---

## What You Can Explore

- Clean accuracy vs robust (adversarial-transport) accuracy
- Effect of transport budget r on robustness
- Dual variable behavior over training
- Inner adversary sensitivity (steps, step size)
- Decision boundary visualization (2D case)

---

## Repository Structure

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

## Local Run

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

---

## Deployment

Railway:
Docker-based deployment using dynamic PORT binding.

Hugging Face Spaces:
Docker-based deployment compatible with CPU instances.

---

## License

MIT License

---

## Citation 

Haque, S. (2026). Wasserstein Strategic Classification (W2 WDRO). Interactive robust learning framework.











