import torch
import torch.nn.functional as F

def wdro_inner_max_w2(
    model,
    x0,                 # (b,d) in [0,1]
    y,                  # (b,1) in {0,1}
    lam_dual: float,    # λ >= 0 (dual weight)
    steps: int,
    step_size: float,
    mutable_mask,       # (d,)
    clamp_box: bool = True,
):
    """
    Inner maximization for W2-Wasserstein DRO (Kantorovich dual inner sup):
        max_{x in constraints}  loss_theta(x,y) - lam * ||x - x0||^2

    Projected gradient ascent on x with:
      - box constraint [0,1]
      - immutability mask
    """
    mask = mutable_mask.view(1, -1)
    x = x0.clone()

    for _ in range(int(steps)):
        x.requires_grad_(True)

        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        cost = ((x - x0) ** 2).sum(dim=1).mean()  # mean transport cost
        obj = loss - float(lam_dual) * cost       # maximize

        g = torch.autograd.grad(obj, x, create_graph=False)[0]
        g = g * mask  # freeze immutable features

        x = (x + float(step_size) * g).detach()

        if clamp_box:
            x = torch.clamp(x, 0.0, 1.0)

        # enforce immutables exactly
        x = x * mask + x0 * (1.0 - mask)

    return x
