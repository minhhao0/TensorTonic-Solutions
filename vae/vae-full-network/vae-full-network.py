import numpy as np

class VAE:
    def __init__(self, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray):
        """
        Initialize VAE with concrete weight matrices.
        """
        # Store weights here
        self.W_mu=np.array(W_mu)
        self.b_mu=np.array(b_mu)
        self.W_logvar=np.array(W_logvar)
        self.b_logvar=np.array(b_logvar)
        self.W_dec=np.array(W_dec)
        self.b_dec=np.array(b_dec)
    def encoder(self,x):
        mu=x@self.W_mu+self.b_mu
        log_var=x@self.W_logvar+self.b_logvar
        return mu,log_var
    def forward(self, x: np.ndarray, epsilon: np.ndarray) -> dict:
        """
        Full forward pass: encode -> reparameterize -> decode.
        Returns dict with "recon", "mu", "log_var".
        """
        # Your implementation here
        mu,log_var=self.encoder(x)
        std=np.exp(0.5*log_var)
        z=mu+std*epsilon
        recon=z@self.W_dec+self.b_dec
        return {
            "recon":recon,
            "mu":mu,
            "log_var":log_var
        }
        
    
    def generate(self, z: np.ndarray) -> np.ndarray:
        """
        Generate samples from given latent vectors.
        """
        # Your implementation here
        return z@self.W_dec+self.b_dec
