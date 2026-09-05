import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = np.max(z)
        exp_values = np.exp(z - max_z)
        sum_of_z = np.sum(exp_values)
        result = ( exp_values / sum_of_z ) % 10
        return np.round(result, 4)

        
        pass
