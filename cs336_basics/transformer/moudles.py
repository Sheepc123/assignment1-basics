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