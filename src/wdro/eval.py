import torch
from src.wdro.adversary import wdro_inner_max_w2

def eval_wdro_metrics(
    model,
    X,
    y,
    mutable_mask,
    lam_dual: float,
    inner_steps: int,
    inner_step_size: float,
):
    """
    Evaluate clean accuracy and WDRO-adversarial accuracy.

    IMPORTANT: the inner adversary requires gradients w.r.t. X,
    so we must ENABLE grads for that part, even during evaluation.
    """
    model.eval()

    # ---- clean metrics (no grads needed) ----
    with torch.no_grad():
        logits = model(X)
        p = torch.sigmoid(logits)
        yhat = (p >= 0.5).float()
        acc_clean = (yhat == y).float().mean().item()

    # ---- adversarial transported points (needs grads!) ----
    with torch.enable_grad():
        X_adv = wdro_inner_max_w2(
            model=model,
            x0=X,
            y=y,
            lam_dual=float(lam_dual),
            steps=int(inner_steps),
            step_size=float(inner_step_size),
            mutable_mask=mutable_mask,
        )

    # ---- adversarial metrics (no grads needed) ----
    with torch.no_grad():
        logits_adv = model(X_adv)
        p_adv = torch.sigmoid(logits_adv)
        yhat_adv = (p_adv >= 0.5).float()
        acc_adv = (yhat_adv == y).float().mean().item()

        avg_l2 = torch.linalg.norm((X_adv - X), dim=1).mean().item()
        avg_cost_sq = ((X_adv - X) ** 2).sum(dim=1).mean().item()

    return {
        "acc_clean": float(acc_clean),
        "acc_wdro_adv": float(acc_adv),
        "avg_l2": float(avg_l2),
        "avg_cost_sq": float(avg_cost_sq),
    }

