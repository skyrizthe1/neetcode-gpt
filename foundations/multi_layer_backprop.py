import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # forward pass
        z1 = np.dot(x, W1.T) + b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2.T) + b2
        loss = np.mean((z2 - y_true) ** 2)

        # backward pass
        dz2 = 2 * (z2 - y_true) / len(y_true)
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        da1 = np.dot(dz2, W2)
        relu_mask = (z1 > 0).astype(float)
        dz1 = da1 * relu_mask
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }