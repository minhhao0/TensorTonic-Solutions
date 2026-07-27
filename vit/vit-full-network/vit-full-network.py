import numpy as np
def softmax(x,axis=-1):
    e_x=np.exp(x-np.max(x,axis=axis,keepdims=True))
    return e_x/np.sum(e_x,axis=axis,keepdims=True)
def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.pow(x,3))))
def layernorm(x,gamma=1,beta=0,epsilon=1e-6):
    mean=np.mean(x,axis=-1,keepdims=True)
    variance=np.var(x,axis=-1,keepdims=True)
    x=gamma*(x-mean)/np.sqrt(variance+epsilon)+beta
    return x
class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.W1=np.array(encoder_weights[0]['W1'])
        self.W2=np.array(encoder_weights[0]['W2'])
        self.Wk=np.array(encoder_weights[0]['Wk'])
        self.Wo=np.array(encoder_weights[0]['Wo'])
        self.Wq=np.array(encoder_weights[0]['Wq'])
        self.Wv=np.array(encoder_weights[0]['Wv'])
        # Initialize weights here
        if W_patch is None:
            W_patch=np.random.randn(3*self.path_size**2,embed_dim)
        self.W_patch=np.array(W_patch)
        self.W_head=np.array(W_head)
        self.pos_embed=np.array(pos_embed)
        self.cls_token=np.array(cls_token)
    def batch_embedding(self,x):
        b,h,w,c=x.shape
        patches=x.reshape(b,h//self.patch_size,self.patch_size,w//self.patch_size,self.patch_size,c).swapaxes(2,3)
        patches=patches.reshape(b,self.num_patches,c*self.patch_size*self.patch_size)
        z=patches@self.W_patch
        return z
    def position_embedding(self,z):
        b=z.shape[0]
        cls_token=np.tile(self.cls_token,(z.shape[0],1,1))
        z=np.concatenate((cls_token,z),axis=1)
        z=z+self.pos_embed
        return z
    def mlp(self,x):
        z=x@self.W1
        y=gelu(z)@self.W2
        return y
    def msa(self,x):
        Q=x@self.Wq
        K=x@self.Wk
        V=x@self.Wv
        d_k=x.shape[1]/self.num_heads
        Qs=np.hsplit(Q,self.num_heads)
        Ks=np.hsplit(K,self.num_heads)
        Vs=np.hsplit(V,self.num_heads)
        h=[]
        for i in range(self.num_heads):
            q=Qs[i]
            k=Ks[i]
            v=Vs[i]
            s=q@k.T
            s=s/np.sqrt(d_k)
            s=softmax(s)
            s=s@v
            h.append(s)
        output=np.concatenate(h,axis=-1)
        output=output@self.Wo
        return output
    def encoder_block(self,x):
        batch_size=x.shape[0]
        result=[]
        for i in range(batch_size):
            x_i=x[i]
            #1st layer norm
            x_1=layernorm(x_i)
            #multi head attention
            mha=self.msa(x_1)
            #1st residual connection
            x_2=x_i+mha
            #2nd layer norm 
            x_3=layernorm(x_2)
            #fff network
            x_4=self.mlp(x_3)
            #2nd residual connection
            x_4=x_2+x_4
            x_4=np.expand_dims(x_4,axis=0)
            result.append(x_4)
        return np.concatenate(result)
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        # YOUR CODE HERE
        x=np.array(x)
        z=self.batch_embedding(x)
        z=self.position_embedding(z)
        z_l=self.encoder_block(z)
        cls=z_l[:,0,:]
        logits=layernorm(cls)
        y=logits@self.W_head
        return y
        