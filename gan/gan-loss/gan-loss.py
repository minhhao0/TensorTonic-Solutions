import numpy as np
def clip(x,eps=10e-9):
    if x <=eps:
        return eps
    if x>=1-eps:
        return 1-eps
    return x
def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    v_clip=np.vectorize(clip)
    real_probs=v_clip(np.array(real_probs))
    fake_probs=v_clip(np.array(fake_probs))
    return -np.mean(np.log(real_probs)+np.log(1-fake_probs))

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    v_clip=np.vectorize(clip)
    fake_probs=v_clip(np.array(fake_probs))
    return -np.mean(np.log(fake_probs))