from jaxtyping import Bool, Float, Int
import einx
import torch
from torch import Tensor


def cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    # l_i = - log softmax(o_i)[x_{i+1}]
    o_max = einx.max("b... [d]", inputs)[0]
    inputs_sub = einx.subtract(
        "b... d, b... -> b... d",
        inputs,
        o_max,
    )

    numerator = -einx.get_at("b... [d], b...  -> b... ", inputs_sub, targets)
    exp_inputs_sub = torch.exp(inputs_sub)
    sum_denom = einx.sum("b... [d]-> b...", exp_inputs_sub)
    denominator = torch.log(sum_denom)
    loss = numerator + denominator
    mean_loss = einx.mean("[b...]", loss)
    return mean_loss
