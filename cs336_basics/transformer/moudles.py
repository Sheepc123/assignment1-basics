from typing import Any

from jaxtyping import Float, Int

from torch import Tensor
import torch
import torch.nn as nn
from einops import rearrange, einsum


# Liner Moudles
class Liner_Module(nn.Module) :
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        
        self.w = nn.Parameter(torch.empty(self.d_out,self.d_in,device=device,
    dtype=dtype))
        
        std = (2 / (in_features + out_features))**0.5
        
        torch.nn.init.trunc_normal_(self.w, mean=0, std= std, a = -3 * std,b = 3 *std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        y = einsum(self.w,x,"d_out d_in,... d_in -> ... d_out")
        return y
    
    
# embedding moudle
class Embedding_moudle(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.e_dim = embedding_dim
        
        ep = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,
    dtype=dtype))
        
        self.w = torch.nn.init.trunc_normal_(ep,0,1,-3,3)
        
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        result = self.w[token_ids]
        return result
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.in_dtype = dtype
        self.device = device
        self.g = nn.Parameter(torch.ones(d_model,dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        
        RMS = (x.pow(2).mean(dim = -1,keepdim=True) + self.eps) ** 0.5 
        
        result = x / RMS * self.g
        
        return result.to(in_dtype)
    
class swiglu(nn.Module):
    def __init__(self, d_model:int ,d_ff: int, device=None, dtype=None):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Liner_Module(d_model,d_ff,device,dtype)
        self.w2 = Liner_Module(d_ff,d_model,device,dtype)
        self.w3 = Liner_Module(d_model,d_ff,device,dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
          
       a = self.w1(x)
       a = a * torch.sigmoid(a)
       b = self.w3(x)
       
       return self.w2(torch.mul(a,b))
   

class rope(nn.Module):
   
    def __init__(self, theta, d_k, max_seq_len, device=None):
        
        super().__init__()
        
        assert d_k % 2 == 0
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        positions =  rearrange(torch.arange(max_seq_len,device=device,dtype=torch.float32,),"n ->n 1")
        
        k = torch.arange(d_k // 2 , device=device,dtype=torch.float32,)
        
        angles =  rearrange(theta**(-1 * 2*k/d_k),"n ->1 n")
         
        rotation_angles = positions@ angles
        
        self.register_buffer("cos_cache",torch.cos(rotation_angles),persistent=False)
        self.register_buffer("sin_cache",torch.sin(rotation_angles),persistent=False)

       
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        in_type = x.dtype
        x = x.to(torch.float32)
        
        x_pair = rearrange(x,"... (d_pair two) -> ... d_pair two",two = 2)
        
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x1 = x_pair[...,-2]
        x2 = x_pair[...,-1]
        
        y1 = x1*cos - sin*x2
        y2 = x1*sin + x2*cos
        
        y = torch.stack((y1,y2),dim = -1)
        
        return rearrange(y, "... d_k1 d_k2 -> ... (d_k1 d_k2)").to(in_type)
        
def softmax(x: torch.Tensor ,dim:int) -> torch.Tensor:
    
    max = x.max(dim = dim, keepdim= True).values
    
    x_stable =  x - max
    x_exp = torch.exp(x_stable)
    
    return x_exp / x_exp.sum(dim=dim,keepdim=True)


def scaled_dot_product(
    q : torch.Tensor, 
    k: torch.Tensor, 
    v : torch.Tensor,
    mask : torch.Tensor | None = None,
    ) -> torch.Tensor:
    
    temp:torch.Tensor = einsum(q,k,"... q_len d_k, ... k_len d_k -> ... q_len k_len" )
    
    d_k = q.shape[-1]
    
    scores : torch.Tensor = temp / (d_k ** 0.5)
    
    
    
    if mask is not None: 
       scores = scores.masked_fill(mask= ~mask,value=float("-inf"))
    
    return softmax(scores,-1)@v

class mt_self_attention(nn.Module):
    
    def  __init__(self, 
                  d_model : int, 
                  num_heads : int,  
                  device=None, 
                  dtype=None,
                  theta = None, 
                  max_seq_len = None,
                  token_positions: Tensor |None = None,
                ):
        
        super().__init__()
        
        assert num_heads > 0
        assert d_model % num_heads == 0
        
        self.single_d =  d_model// num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.rope = None
        
        self.Q_w = Liner_Module(d_model,d_model,device,dtype)
        self.K_w = Liner_Module(d_model,d_model,device,dtype)
        self.V_w = Liner_Module(d_model,d_model,device,dtype)
        
        # # the optimization funciton
        # self.qkv_w = Liner_Module(d_model*3,d_model)
        
        
        self.O_w = Liner_Module(d_model,d_model,device,dtype)
        self.token_poisitions = token_positions
        self.theta = theta
        
        
        # 
        
        if theta is not None :
            assert max_seq_len is not None
            assert self.single_d % 2 == 0
            
            self.rope = rope(theta=theta, d_k = self.single_d, max_seq_len=max_seq_len)
                
    
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        q = self.Q_w(x)
        k=  self.K_w(x)
        v = self.V_w(x)
        
        # qkv = self.qkv_w(x)
        # q = qkv[..., :self.d_model]
        # k = qkv[..., self.d_model:self.d_model *2]
        # v = qkv[..., self.d_model *2:]
      
        
   
        
        
        q_pairs = rearrange(q,"... T (h d_k) -> ... h T d_k",h = self.num_heads)
        K_pairs = rearrange(k,"... T (h d_k) -> ... h T d_k",h = self.num_heads)
        v_pairs = rearrange(v,"... T (h d_k) -> ... h T d_k",h = self.num_heads)
        
        seq_len = x.shape[-2]
      
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        
        if self.rope is not None:
            if self.token_poisitions is None:
                positions = torch.arange(
                    seq_len,
                    device=x.device,
                    dtype=torch.long
                )
            else:
                positions = self.token_poisitions
                
                
            q_pairs = self.rope(q_pairs,positions)
            K_pairs = self.rope(K_pairs,positions)
        
        ans = scaled_dot_product(q_pairs,K_pairs,v_pairs,mask)
        
        ans = rearrange(ans,"... h t d_k -> ... t (h d_k)")
        
        output = self.O_w(ans)
        
        return output
       
    
        
class Transformer_block(nn.Module):
    def __init__(self,d_model,d_ff,num_heads,theta,max_sequence_len,device = None,dtype  = None):
        super().__init__()
        
        self.RMSNorm1 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.RMSNorm2 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        
        self.mt_self_attention = mt_self_attention(
                                d_model=d_model,
                                device=device,
                                dtype=dtype,
                                num_heads=num_heads,max_seq_len=max_sequence_len,
                                theta=theta,
                                )
        self.swiglu = swiglu(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
            )
        
    def forward(self, x : torch.Tensor) ->torch.Tensor:
        
        # phase1 
        y = x + self.mt_self_attention(self.RMSNorm1(x))
       
        # phase 2
        y = y + self.swiglu(self.RMSNorm2(y))
        
        return y
        
        
from cs336_basics.bpe import tokenization2 as tokenizaiton         
        
class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        dtype = None,
        device = None,
    ):
        super().__init__()
        
        self.embedding = Embedding_moudle(vocab_size,d_model, device=device,
    dtype=dtype)
        
        self.layers = nn.ModuleList([
            Transformer_block(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                theta=rope_theta,
                max_sequence_len= context_length,
                 device=device,
                dtype=dtype
            ) for i in range(num_layers)
        ])
        
        self.vocab_size = vocab_size
        
        self.FNorm = RMSNorm(d_model=d_model, device=device,dtype=dtype)
        self.FLiner = Liner_Module(d_model,vocab_size, device=device, dtype=dtype)
        
    def forward(self,indice:torch.Tensor) -> torch.Tensor:
      
        output = self.embedding(indice)
        
        for layer in self.layers:
            output = layer(output)
        
        output = self.FNorm(output)
        output = self.FLiner(output)
        
        return output
        
        
        
        