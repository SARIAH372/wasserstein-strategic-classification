import numpy as np
import torch
import torch.nn.functional as F

from src.utils import iter_minibatches


def train_erm(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
):
    """
    Clean ERM training with Adam on BCE-with-logits.

    Args:
      model: torch nn.Module returning logits (shape [b,1])
      X: torch tensor [n,d]
      y: torch tensor [n,1] with values in {0,1}
    Returns:
      hist: list of dicts with epoch and loss
    """
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    hist = []
    n = X.shape[0]

    for ep in range(1, int(epochs) + 1):
        losses = []
        for idx in iter_minibatches(n=int(n), batch_size=int(batch_size), seed=int(seed) + ep):
            xb = X[idx]
            yb = y[idx]

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        hist.append({"epoch": ep, "loss": float(np.mean(losses))})

    return hist
