import numpy as np
import torch
import torch.nn.functional as F
from src.utils import iter_minibatches
from src.wdro.adversary import wdro_inner_max_w2

def train_wdro_w2_dual(
    model,
    X,
    y,
    mutable_mask,
    rho: float,
    lam_init: float,
    eta_lam: float,
    inner_steps: int,
    inner_step_size: float,
    lr: float,
    epochs: int,
    batch_size: int,
    mix_clean_frac: float,
    seed: int = 7,
):
    """
    W2 WDRO dual training:
      min_theta  lam*rho + E[ sup_x ( loss(x) - lam||x-x0||^2 ) ]
    lam update:
      lam <- max(0, lam + eta*(E[cost]-rho))
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lam = float(max(lam_init, 0.0))
    hist = []

    n = X.shape[0]
    for ep in range(1, epochs + 1):
        losses = []
        costs = []

        for idx in iter_minibatches(n, batch_size, seed + 5000 + ep):
            xb = X[idx]
            yb = y[idx]

            opt.zero_grad(set_to_none=True)

            x_star = wdro_inner_max_w2(
                model=model,
                x0=xb,
                y=yb,
                lam_dual=lam,
                steps=int(inner_steps),
                step_size=float(inner_step_size),
                mutable_mask=mutable_mask,
            )

            # mix clean for stability
            if mix_clean_frac > 0:
                k = int(np.floor(mix_clean_frac * xb.shape[0]))
                if k > 0:
                    X_mix = torch.cat([x_star, xb[:k]], dim=0)
                    y_mix = torch.cat([yb, yb[:k]], dim=0)
                else:
                    X_mix, y_mix = x_star, yb
            else:
                X_mix, y_mix = x_star, yb

            loss = F.binary_cross_entropy_with_logits(model(X_mix), y_mix)
            total = loss + lam * float(rho)  # constant in theta but part of objective

            total.backward()
            opt.step()

            with torch.no_grad():
                cost_sq = ((x_star - xb) ** 2).sum(dim=1).mean().item()

            losses.append(float(total.item()))
            costs.append(float(cost_sq))

        avg_cost_sq = float(np.mean(costs)) if costs else 0.0
        lam = max(0.0, lam + float(eta_lam) * (avg_cost_sq - float(rho)))

        hist.append({
            "epoch": ep,
            "loss_total": float(np.mean(losses)) if losses else 0.0,
            "avg_cost_sq": avg_cost_sq,
            "rho": float(rho),
            "lambda_dual": float(lam),
        })

    return hist
