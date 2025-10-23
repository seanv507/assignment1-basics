import torch
import einx 


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int,eps: float=1e-5, 
                 device: torch.device | None=None, dtype: torch.dtype|None=None)-> None:
        super().__init__()
        # subclass nn.Module
        # call the superclass constructor
        self.d_model = d_model
        self.eps = eps 
        factory_kwargs = {"device": device, "dtype": dtype}
        self.g = torch.nn.Parameter(torch.empty((d_model,), **factory_kwargs))

    def reset_parameters(self):
        torch.nn.init.ones_(self.g) 
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """ given x  and weights """       
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # sqrt(1/d_model) \sum_{i=1} {d_model} a_i^2 + \epsilon
        summand = einx.elementwise("a... -> a...", x, op=torch.square)
        summand = einx.add("a... -> a...", summand, self.eps)
        rms = einx.sum(" b... [d]", summand)
        rms = einx.elementwise("a... -> a...", rms,self.d_model, op=torch.divide)
        rms = einx.elementwise("a... -> a...", rms, op=torch.sqrt)
        print(x.size(), rms.size(), self.g.size())
        out = einx.divide("a... b, a... -> a... b", x, rms)
        out = einx.multiply("a... b, b -> a... b", out, self.g)
        # , * self.g
        out = out.to(in_dtype)
        return out
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"
    