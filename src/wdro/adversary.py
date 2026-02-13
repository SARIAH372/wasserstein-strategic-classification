import torch
import torch.nn.functional as F


def wdro_inner_max_w2(
    model,
    x0: torch.Tensor,          # (b,d) in [0,1]
    y: torch.Tensor,           # (b,1) in {0,1}
    lam_dual: float,           # lambda >= 0
    steps: int,
    step_size: float,
    mutable_mask: torch.Tensor # (d,) float {0,1}
) -> torch.Tensor:
    """
    Inner maximization (Kantorovich dual inner sup) for W2-WDRO:

      x_adv = argmax_x  BCEWithLogits(model(x), y) - lam * mean(||x-x0||^2)

    constraints:
      - box: x in [0,1]^d
      - immutability: x_j = x0_j for mask_j = 0

    Implementation notes:
      - Requires gradients w.r.t. x.
      - Uses projected gradient ascent.
      - Adds a tiny random jitter on mutable dims to avoid flat-start traps.
      - Returns a detached tensor.
    """
    model.eval()

    mask = mutable_mask.view(1, -1).to(dtype=x0.dtype, device=x0.device)
    x0_det = x0.detach()

    # start at x0
    x = x0_det.clone()

    # ---- NEW: tiny random jitter on mutable dims to avoid zero-gradient / flat-start traps
    if float(mask.sum().item()) > 0.0:
        x = x + 0.01 * torch.randn_like(x) * mask
        x = torch.clamp(x, 0.0, 1.0)
        x = x * mask + x0_det * (1.0 - mask)
    # -------------------------------------------------------------------------------

    lam = float(max(lam_dual, 0.0))
    eta = float(step_size)

    for _ in range(int(steps)):
        x.requires_grad_(True)

        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        # mean squared transport cost
        cost = ((x - x0_det) ** 2).sum(dim=1).mean()

        # maximize objective
        obj = loss - lam * cost

        # gradient wrt x
        g = torch.autograd.grad(obj, x, create_graph=False, retain_graph=False)[0]

        # apply immutability: only mutable dims can move
        g = g * mask

        # ascent step
        x = (x + eta * g).detach()

        # project to box
        x = torch.clamp(x, 0.0, 1.0)

        # enforce immutables exactly
        x = x * mask + x0_det * (1.0 - mask)

    return x.detach()
