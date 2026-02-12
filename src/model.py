import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits

