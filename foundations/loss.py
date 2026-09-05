import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred , 1e-7 , 1 - 1e-7)

        n = len(y_true)

        multiply = ( y_true * np.log(y_pred) ) + ( (1 - y_true) * ( np.log(1 - y_pred) ) )

        sum_of_all = -1/n * np.sum(multiply)
        return np.round(sum_of_all , 4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred , 1e-7 , 1 - 1e-7)
        n = len(y_true)
        multiplys = np.sum( y_true * np.log(y_pred) )
        final = -1/n * np.sum(multiplys)
        return np.round(final , 4)

        pass
