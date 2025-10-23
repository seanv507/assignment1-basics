import torch
import einx

class SoftMax(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    
    def forward(self, v):
        v_max = einx.max("b... [d] c...", v,d=5, graph=True) #we put in size of dimension
        #  b=v.shape[:self.dim]
        v_max = v.max(self.dim, keepdim=True)[0]
        tmp = v -v_max
        tmp = torch.exp_(tmp)
        tmp /= tmp.sum(axis=self.dim, keepdim=True)

       
        # how to pass in named parameter
        # v_max = einx.max("b... [d]", v, d=5, graph=True) we put in size of dimension
        # tmp = einx.subtract("b... [d] c..., b... c...-> b... d c...", v, v_max, d=dim)
        # tmp = einx.elementwise("b... -> b...", tmp, op=torch.exp)
        # sum_exp = einx.sum("b... [d] c...-> b... c...", tmp, d=dim)
        # out = einx.divide("b... [d] c...-> b... d c...", tmp, sum_exp, d=dim)
        return tmp
    