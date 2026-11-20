import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)
        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> dict:
        """
        Returns outputs and final_hidden_state as float64 arrays.
        """

        batch_size=X.shape[0]
        output=[]
        hidden_states=[]
        T = X.shape[1]
        for batch in range(batch_size):
            x=X[batch]
            h_0=np.zeros(self.hidden_dim)
            hs=[h_0]
            ys = []
            for t in range(T):
                x_t=x[t]
                h_prev=hs[t]
                z_t=sigmoid(np.concatenate([h_prev,x_t],axis=-1)@self.W_z.T+self.b_z)
                r_t=sigmoid(np.concatenate([h_prev,x_t],axis=-1)@self.W_r.T+self.b_r)
                h_t=np.tanh(np.concatenate([r_t*h_prev,x_t],axis=-1)@self.W_h.T+self.b_h)
                h=z_t*h_prev+(1-z_t)*h_t
                y=h@self.W_y.T+self.b_y
                hs.append(h)
                ys.append(y)
            output.append(ys)
            hidden_states.append(hs)
        output=np.stack(output)
        hidden_states=np.stack(hidden_states)
        return {"outputs":output,"final_hidden_state":hidden_states[:,-1,:]}