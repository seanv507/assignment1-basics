import math
import torch
import einx



class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, device: torch.device|None=None, dtype:torch.dtype|None=None):
        super().__init__()
        self.d_model=d_model

        if not d_ff: 
            d_ff = int(math.ceil((8/3 * d_model) /64))*64
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight_1 = torch.nn.Parameter(torch.empty((d_ff, d_model, ), **factory_kwargs))
        self.weight_3 = torch.nn.Parameter(torch.empty((d_ff, d_model, ), **factory_kwargs))
        self.weight_2 = torch.nn.Parameter(torch.empty((d_model, d_ff, ), **factory_kwargs))
        
        self.reset_parameters()

    def reset_parameters(self):
        std = (2/(self.weight_1.size(0) + self.weight_1.size(1)))**0.5
        torch.nn.init.trunc_normal_(self.weight_1, std=std,a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.weight_2, std=std,a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.weight_3, std=std,a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        w1_x = einx.dot("f [d], b... [d] -> b... f", self.weight_1, x)
        tmp = einx.elementwise("b...  ->b... ", w1_x, op=torch.sigmoid)
        silu = einx.multiply("b..., b... -> b... ",w1_x, tmp) #silu

        w3_x = einx.dot("f [d], b... [d] -> b... f", self.weight_3, x)
        tmp = einx.multiply("b..., b... -> b...",silu, w3_x)
        out = einx.dot("d [f], b... [f] -> b... d", self.weight_2, tmp)
        return out