import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def clip(x):
    if x<=10e-9:
        return 10e-9
    if x>=1-10e-9:
        return 1-10e-9
    return x
def discriminator_loss(p_r,p_f):
    v_clip=np.vectorize(clip)
    p_r=v_clip(p_r)
    p_f=v_clip(p_f)
    return -np.mean(np.log(p_r)+np.log(1-p_f))
def generator_loss(p_f):
    v_clip=np.vectorize(clip)
    p_f=v_clip(p_f)
    return -np.mean(np.log(p_f))
def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    # Your implementation here
    real_data=np.array(real_data)
    fake_data=np.array(fake_data)
    D_W=np.array(D_W)
    p_r=[]
    p_f=[]
    n_batch=real_data.shape[0]
    for i in range(n_batch):
        y=sigmoid(real_data[i]@D_W)
        y1=sigmoid(fake_data[i]@D_W)
        p_r.append(y)
        p_f.append(y1)
    p_r=np.array(p_r)
    p_f=np.array(p_f)
    return {
        "d_loss":discriminator_loss(p_r,p_f),
         "g_loss":generator_loss(p_f)
    }
    