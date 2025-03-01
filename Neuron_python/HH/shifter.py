import numpy as np
########################  量化函数   ########################

def quantize_in(x, int_bits=4, frac_bits=16):
    sign = np.sign(x)
    qx = np.floor(x * 2 ** frac_bits) / 2 ** frac_bits
    qx = np.clip(np.abs(qx), 0, 2 ** int_bits)
    qx = qx * sign
    return qx
def quantize_out(x, int_bits=6, frac_bits=16):
    sign = np.sign(x)
    qx = np.floor(x * 2 ** frac_bits) / 2 ** frac_bits
    qx = np.clip(np.abs(qx), 0, 2 ** int_bits)
    qx = qx * sign
    return qx

def shifter_4_16(x,n):
    x = x * (2**-n)
    result = quantize_in(x)
    return result

def shifter_6_16(x,n):
    x = x * (2**-n)
    result = quantize_out(x)
    return result