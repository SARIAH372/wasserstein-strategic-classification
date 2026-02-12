import numpy as np

def _corr_gauss(n: int, d: int, rho: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(d)
    C = rho ** np.abs(np.subtract.outer(idx, idx))
    L = np.linalg.cholesky(C + 1e-8 * np.eye(d))
    Z = rng.normal(size=(n, d))
    return Z @ L.T

def make_synthetic(n: int, d: int, seed: int, rho: float, nonlinear: bool, label_noise: float):
    """
    Returns:
      X in [0,1]^d
      y in {0,1}
    """
    Xg = _corr_gauss(n, d, rho, seed)
    X = 1 / (1 + np.exp(-Xg))  # map to [0,1]

    rng = np.random.default_rng(seed + 321)
    w = rng.normal(size=d)
    w = w / (np.linalg.norm(w) + 1e-12)

    if nonlinear and d >= 2:
        s = X @ w + 0.55 * (X[:, 0] * X[:, 1]) + 0.15 * np.sin(7 * X[:, 0]) - 0.10 * np.cos(5 * X[:, 1])
    else:
        s = X @ w

    p = 1 / (1 + np.exp(-3 * s))
    y = (rng.random(n) < p).astype(int)

    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y[flip] = 1 - y[flip]

    return X.astype(np.float32), y.astype(int)
