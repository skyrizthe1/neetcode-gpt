import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights = np.array(initial_weights, dtype=np.float64)
        N = len(X)
        
        for i in range(num_iterations):
            prediction = self.get_model_prediction(X , weights)

            d0 = self.get_derivative(prediction , Y , N , X , 0)
            d1 = self.get_derivative(prediction , Y , N , X , 1)
            d2 = self.get_derivative(prediction , Y , N , X , 2)

            gradient = np.array([d0 , d1 , d2])
            weights = weights - self.learning_rate * gradient

        return np.round(weights, 5)
        pass
