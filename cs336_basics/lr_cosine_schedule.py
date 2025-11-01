import math


def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        alpha_t = t / T_w * alpha_max
    elif t <= T_c:
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min

    return alpha_t
