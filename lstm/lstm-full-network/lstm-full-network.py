import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last, C_last).
        """
        # YOUR CODE HERE
        h_0=np.zeros((X.shape[0],self.hidden_dim))
        C_0=np.zeros((X.shape[0],self.hidden_dim))
        hidden_states=[h_0]
        cell_state=[C_0] 
        y=[]
        for t in range(1,X.shape[1]+1):
            x_t=X[:,t-1,:]
            h_prev=hidden_states[t-1]
            C_prev=cell_state[t-1]
            concat=np.concatenate([h_prev,x_t],axis=-1)
            #forget gate
            f_t=sigmoid(concat@self.W_f.T+self.b_f)
            #input gate 
            i_t=sigmoid(concat@self.W_i.T+self.b_i)
            #candidate cell state
            c_t=np.tanh(concat@self.W_c.T+self.b_c)
            #cell state update
            C_t=f_t*C_prev+i_t*c_t
            #output gate
            o_t=sigmoid(concat@self.W_o.T+self.b_o)
            #hidden state
            h_t=o_t*np.tanh(C_t)
            hidden_states.append(h_t)
            y_t=h_t@self.W_y.T+self.b_y
            y.append(y_t)
            cell_state.append(C_t)
        hidden_states=np.stack(hidden_states,axis=1)
        y=np.stack(y,axis=1)
        cell_state=np.stack(cell_state,axis=1)
        return y,hidden_states[:,-1,:],cell_state[:,-1,:]