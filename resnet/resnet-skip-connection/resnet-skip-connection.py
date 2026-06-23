import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # YOUR CODE HERE
    result=x
    for i in range(len(gradients_F)):
        grad=np.array(gradients_F[i])
        result=result+result@grad
    return result
    
def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    result=x
    for i in range(len(gradients_F)):
        grad=np.array(gradients_F[i])
        result=result@grad
    return result
