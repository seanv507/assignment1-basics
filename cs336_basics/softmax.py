import torch
import einx

class SoftMax(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, v, dimension:int):
        
        # b=v.shape[:self.dimension] #tuple to identify d
        # v_max = einx.max("b... [d] c...", v, b=b) #we put in size of dimension
        # tmp = einx.subtract("b... [d] c..., b... c...-> b... d c...", v, v_max, b = b, d=1, c=0)
        # tmp = einx.elementwise("b... -> b...", tmp, op=torch.exp)
        # sum_exp = einx.sum("b... [d] c...-> b... c...", tmp, b=b)
        # out = einx.divide("b... [d] c...-> b... d c...", tmp, sum_exp, b=b)

        v_max = v.max(dimension, keepdim=True)[0]
        out = v -v_max
        out = torch.exp_(out)
        out /= out.sum(axis=dimension, keepdim=True)

       
        return out
    