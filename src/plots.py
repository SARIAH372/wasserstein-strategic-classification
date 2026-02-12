import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_curve(x, y, xlabel: str, ylabel: str, title: str):
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return fig

def plot_hist_cost_l2(costs, title="||Δ||2 histogram"):
    fig = plt.figure()
    plt.hist(costs, bins=30)
    plt.xlabel("||x' - x||2")
    plt.ylabel("count")
    plt.title(title)
    return fig

def plot_decision_2d(X_np, y_np, model, title="Decision surface"):
    if X_np.shape[1] != 2:
        fig = plt.figure()
        plt.text(0.05, 0.5, "2D plot available only when d=2")
        plt.axis("off")
        return fig

    x1 = np.linspace(0, 1, 250)
    x2 = np.linspace(0, 1, 250)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(grid))
        p = torch.sigmoid(logits).view(xx.shape).cpu().numpy()

    fig = plt.figure()
    plt.contourf(xx, yy, p, levels=25)
    plt.scatter(X_np[:, 0], X_np[:, 1], s=12, c=y_np)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    return fig
