import numpy as np
def get_alpha_bar(betas):
    result=[]
    product=1
    for beta in betas:
        product*=(1-beta)
        result.append(product)
    return result
def ddpm_sample(x_T: list, betas: list[float], epsilon_preds: list, z_values: list) -> list:
    """
    Returns the final denoised sample rounded to four decimals.
    """
    T=len(betas)
    x_T=np.array(x_T)
    z_values=np.array(z_values)
    epsilon_preds=np.array(epsilon_preds)
    alpha_bar=get_alpha_bar(betas)
    for i in range(T):
        t=T-i-1
        alpha_t=1-betas[t]
        mu_t=(1/np.sqrt(alpha_t))*(x_T-(betas[t]/(np.sqrt(1-alpha_bar[t])))*epsilon_preds[i])
        print(mu_t)
        if t==0:
            x_T=mu_t
        else:
            
            x_T=mu_t+np.sqrt(betas[t])*z_values[i]
        print(x_T)
    return x_T
        