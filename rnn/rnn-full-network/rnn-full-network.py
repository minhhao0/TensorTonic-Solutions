import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        y_t=[]
        hidden_states=[h_0]
        for i in range(1,X.shape[1]+1):
            x_t=X[:,i-1,:]
            h_t=np.tanh(x_t@self.W_xh.T+hidden_states[i-1]@self.W_hh.T+self.b_h)
            hidden_states.append(h_t)
            y_t.append(h_t@self.W_hy.T+self.b_y)
        hidden_states=np.stack(hidden_states,axis=1)
        h_final=hidden_states[:,-1,:]
        return np.stack(y_t,axis=1),h_final
            