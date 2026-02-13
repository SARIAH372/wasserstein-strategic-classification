import torch
import torch.nn.functional as F
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
    Returns both accuracy metrics and *sensitive* robustness metrics.
    Accuracy can stay identical even when the attack is active, so we also report:
      - flip_rate
      - loss_clean / loss_adv
      - loss_increase
    """
    model.eval()

    # ---- clean preds + loss (no grad) ----
    with torch.no_grad():
        logits = model(X)
        loss_clean = F.binary_cross_entropy_with_logits(logits, y).item()
        p = torch.sigmoid(logits)
        yhat = (p >= 0.5).float()
        acc_clean = (yhat == y).float().mean().item()

    # ---- adversarial x' (needs grad) ----
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

    # ---- adv preds + loss (no grad) ----
    with torch.no_grad():
        logits_adv = model(X_adv)
        loss_adv = F.binary_cross_entropy_with_logits(logits_adv, y).item()
        p_adv = torch.sigmoid(logits_adv)
        yhat_adv = (p_adv >= 0.5).float()
        acc_adv = (yhat_adv == y).float().mean().item()

        # fraction of predictions changed by the attack
        flip_rate = (yhat_adv != yhat).float().mean().item()

        avg_l2 = torch.linalg.norm((X_adv - X), dim=1).mean().item()
        avg_cost_sq = ((X_adv - X) ** 2).sum(dim=1).mean().item()

    return {
        "acc_clean": float(acc_clean),
        "acc_wdro_adv": float(acc_adv),
        "flip_rate": float(flip_rate),
        "loss_clean": float(loss_clean),
        "loss_adv": float(loss_adv),
        "loss_increase": float(loss_adv - loss_clean),
        "avg_l2": float(avg_l2),
        "avg_cost_sq": float(avg_cost_sq),
    }

