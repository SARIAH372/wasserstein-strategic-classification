import numpy as np

def iter_minibatches(n: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield idx[i:i+batch_size]
