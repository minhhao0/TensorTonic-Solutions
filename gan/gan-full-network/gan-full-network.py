import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def clip(x):
    if x<=10e-9:
        return 10e-9
    if x>=1-10e-9:
       return 1-10e-9
    return x
class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        x=np.array(z)
        return np.tanh(x@self.G_W)
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        x=np.array(x)
        logits=[]
        batches=x.shape[0]
        for i in range(batches):
           logit=sigmoid(x[i]@self.D_W)
           logits.append(logit)
        return logits
    def distriminator_loss(self,p_r,p_f):
        v_clip=np.vectorize(clip)
        p_r=v_clip(p_r)
        p_f=v_clip(p_f)
        return -np.mean(np.log(p_r)+np.log(1-p_f))
    def generator_loss(self,p_f):
        v_clip=np.vectorize(clip)
        p_f=v_clip(p_f)
        return -np.mean(np.log(p_f))
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        # Your implementation here
        real_data=np.array(real_data)
        z=np.array(z)
        z=self.generate(z)
        p_r=self.discriminate(real_data)
        p_f=self.discriminate(z)
        return {
            "d_loss":self.distriminator_loss(p_r,p_f),
            "g_loss":self.generator_loss(p_f)
        }
        