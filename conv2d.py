import torch


def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + w, j:j + h] * K).sum()
    return Y
