import time
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
    max_wall_seconds: int = 240,      # <-- NEW: stop after 4 minutes
    verbose_every: int = 1,           # <-- NEW: log every epoch
):
    """
    W2 WDRO dual training with a wall-time cap.
    Returns partial history if capped (so Streamlit can always display something).
    """
    start = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    lam = float(max(lam_init, 0.0))
    hist = []

    n = X.shape[0]
    model.train()

    for ep in range(1, int(epochs) + 1):
        losses = []
        costs = []

        for idx in iter_minibatches(n=int(n), batch_size=int(batch_size), seed=int(seed) + 5000 + ep):
            xb = X[idx]
            yb = y[idx]

            # inner adversary (needs grads)
            with torch.enable_grad():
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
                k = int(np.floor(float(mix_clean_frac) * xb.shape[0]))
                if k > 0:
                    X_mix = torch.cat([x_star, xb[:k]], dim=0)
                    y_mix = torch.cat([yb, yb[:k]], dim=0)
                else:
                    X_mix, y_mix = x_star, yb
            else:
                X_mix, y_mix = x_star, yb

            opt.zero_grad(set_to_none=True)
            logits = model(X_mix)
            loss = F.binary_cross_entropy_with_logits(logits, y_mix)
            total = loss + lam * float(rho)

            total.backward()
            opt.step()

            with torch.no_grad():
                cost_sq = ((x_star - xb) ** 2).sum(dim=1).mean().item()

            losses.append(float(total.item()))
            costs.append(float(cost_sq))

            # wall-time cap (prevents Railway kill with no output)
            if (time.time() - start) > max_wall_seconds:
                break

        avg_cost_sq = float(np.mean(costs)) if costs else 0.0
        lam = max(0.0, lam + float(eta_lam) * (avg_cost_sq - float(rho)))

        row = {
            "epoch": ep,
            "loss_total": float(np.mean(losses)) if losses else 0.0,
            "avg_cost_sq": avg_cost_sq,
            "rho": float(rho),
            "lambda_dual": float(lam),
            "elapsed_sec": float(time.time() - start),
        }
        hist.append(row)

        if verbose_every and (ep % int(verbose_every) == 0):
            print(f"[v4] ep={ep} loss={row['loss_total']:.4f} cost={row['avg_cost_sq']:.4f} lam={row['lambda_dual']:.3f} t={row['elapsed_sec']:.1f}s")

        if (time.time() - start) > max_wall_seconds:
            break

    return hist
