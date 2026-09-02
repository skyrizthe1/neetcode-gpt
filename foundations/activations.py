import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        n = len(z)
        result = []
        for i in range(n):
             sig = 1 / (1 + np.exp(-z[i] ) )
             rounded_num = round(sig , 5)
             result.append(rounded_num)
            
        return result
    
        

        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        results = []
        for i in range(len(z)):

                results.append(max(0.0 , z[i]))

        return results
                
        pass
