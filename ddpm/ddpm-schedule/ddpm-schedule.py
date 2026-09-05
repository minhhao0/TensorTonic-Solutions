import numpy as np

def linear_beta_schedule(T: int,
                         beta_1: float = 0.0001,
                         beta_T: float = 0.02) -> list[float]:
    """
    Returns T linearly spaced beta values.
    """
    # betas=[beta_1]
    # if T==1:
    #     return betas
    # for i in range(1,T):
    #     t=i+1
    #     beta_t=beta_1+((t-1)/(T-1))*(beta_T-beta_1)
    #     betas.append(round(beta_t,6))
    # return betas
    return np.round(np.linspace(beta_1, beta_T, T), 6).tolist()