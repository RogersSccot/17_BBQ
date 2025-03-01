import numpy as np
import matplotlib.pyplot as plt
from RARCORDIC_version4 import *


# # Tonic spiking
C = 200  # 200pF
gL = 10  # 10nS
El = -70  # -70mV
VT = -50  # -50mV
delt_T = 2  # 2mV
alpha = 2  # 0nS
tw = 30  # 30mS
b = 0  # -58pA
Vr = -58  # 500mV
I = 500  # 2pA

# Tonic bursting
# C = 200  # 200pF
# gL = 12  # 10nS
# El = -60  # -70mV
# VT = -50  # -50mV
# delt_T = 2  # 2mV
# alpha = 2  # 0nS
# tw = 300  # 30mS
# b = 30  # -58pA
# Vr = -46  # 500mV
# I = 400  # 2pA

# adaption
# C = 200  # 200pF
# gL = 12  # 10nS
# El = -70  # -70mV
# VT = -50  # -50mV
# delt_T = 2  # 2mV
# alpha = 2  # 0nS
# tw = 300  # 30mS
# b = 60  # -58pA
# Vr = -58  # 500mV
# I = 500  # 2pA

# initial burst
# C = 130  # 200pF
# gL = 18  # 10nS
# El = -58  # -70mV
# VT = -50  # -50mV
# delt_T = 2  # 2mV
# alpha = 4  # 0nS
# tw = 150  # 30mS
# b = 120  # -58pA
# Vr = -50  # 500mV
# I = 400  # 2pA

# regular bursting
# C = 200  # 200pF
# gL = 10  # 10nS
# El = -58  # -70mV
# VT = -50  # -50mV
# delt_T = 2  # 2mV
# alpha = 2  # 0nS
# tw = 120  # 30mS
# b = 100  # -58pA
# Vr = -46  # 500mV
# I = 400  # 2pA

# delayed accelerating
# C = 200  # 200pF
# gL = 12  # 10nS
# El = -70  # -70mV
# VT = -50  # -50mV
# delt_T = 2  # 2mV
# alpha = -10  # 0nS
# tw = 300  # 30mS
# b = 0  # -58pA
# Vr = -58  # 500mV
# I = 300  # 2pA

# delayed regular bursting
# C = 200  # 200pF
# gL = 10  # 10nS
# El = -65  # -70mV
# VT = -50  # -50mV
# delt_T = 2  # 2mV
# alpha = -10  # 0nS
# tw = 300  # 30mS
# b = 30  # -58pA
# Vr = -46  # 500mV
# I = 110  # 2pA

# transient spiking
# C = 100  # 200pF
# gL = 12  # 10nS
# El = -68  # -70mV
# VT = -50  # -50mV
# delt_T = 2  # 2mV
# alpha = 20  # 0nS
# tw = 90  # 30mS
# b = 350  # -58pA
# Vr = -47  # 500mV
# I = 350  # 2pA

# phasic bursting
# C = 100  # 200pF
# gL = 10  # 10nS
# El = -70  # -70mV
# VT = -50  # -50mV
# delt_T = 2  # 2mV
# alpha = 10  # 0nS
# tw = 90  # 30mS
# b = 30  # -58pA
# Vr = -47  # 500mV
# I = 350  # 2pA

dt = 1/100
T = 45000


# TODO: Generate float point potential data
def GFloatV():
    v = []
    w = []
    exp = []
    v.append(Vr)
    w.append(0)
    spiking_point_float = []
    spiking_time_float = []
    for t in range(T):
        exp_in = (v[t] - VT) / delt_T
        exp.append(exp_in)
        exp_out = np.exp(exp_in)
        x = v[t] + dt * (1 / C) * (-gL * (v[t] - El) + gL * delt_T * exp_out + I - w[t])  # mV
        y = w[t] + dt * (1 / tw) * (alpha * (v[t] - El) - w[t])  # pA
        if x > 0:
            v.append(Vr)
            spiking_point_float.append(Vr)
            w.append(w[t] + b)
            spiking_time_float.append(t)
        else:
            v.append(x)
            spiking_point_float.append(x)
            w.append(y)
    return spiking_point_float, spiking_time_float,w


# TODO: Generate fixed point potential data
def quantize(x, int_bits, frac_bits):
    sign = np.sign(x)
    qx = np.floor(x * 2 ** frac_bits) / 2 ** frac_bits
    qx = np.clip(np.abs(qx), 0, 2 ** int_bits)
    qx = qx * sign
    return qx


def GFixedV():
    # Parameters of Pipeline architecture
    int_bits = 7
    frac_bits = 19


    # shifter = 2 ** 5
    # A1 = quantize((1 - gL * dt / C) , int_bits, frac_bits)
    # A2 = quantize(gL * delt_T * dt / C, int_bits, frac_bits)
    # A3 = quantize((I + gL * El) * dt / C, int_bits, frac_bits)
    # A4 = quantize((-dt / C) * 2 ** 1, int_bits, frac_bits)
    # B1 = quantize((1 - dt / tw), int_bits, frac_bits)
    # B2 = quantize((alpha * dt / tw), int_bits, frac_bits)
    # B3 = quantize(-alpha * El * dt / tw, int_bits, frac_bits)
    A1 = 524025 / 524288
    A2 = 524 / 524288
    A3 = -5243 / 524288
    A4 = -53 / 524288
    B1 = 524113 / 524288
    B2 = 349 / 524288
    B3 = 24466 / 524288

    # exp0 = []
    A011 = []
    A022 = []
    A033 = []
    A044 = []
    B011 = []
    B022 = []
    B033 = []
    v_fixed = []
    w_fixed = []
    v_fixed.append(Vr)
    w_fixed.append(0)
    spiking_point_fixed = []
    spiking_time_fixed = []
    exp =  []
    v_shift_ = []
    cnt_sum = 0



    for t in range(T):
        thres = quantize(v_fixed[t] - VT,int_bits,frac_bits)
        exp_in =  quantize(quantize(quantize(v_fixed[t] - VT,int_bits,frac_bits) / delt_T,int_bits,frac_bits) -452706/2**16, 4, 16)
        # exp_in = quantize((v_fixed[t] - VT) / delt_T + np.log(A2) , int_bits, frac_bits)
        exp_out,cnt = AR_CORDIC_quantize(1 , exp_in, 0)
        A11 = quantize(A1 * v_fixed[t],int_bits,frac_bits)
        A22 = exp_out
        A33 = A3
        A44 = quantize(quantize(A4 * w_fixed[t],int_bits,frac_bits) / 2 ** 2,int_bits,frac_bits)
        x = quantize(quantize(quantize(A11 + A33,int_bits,frac_bits) + A44,int_bits,frac_bits) + A22,int_bits,frac_bits) # mV

        B11 = quantize(quantize(B1 * w_fixed[t] ,int_bits,frac_bits) / 2 ** 0,int_bits,frac_bits)
        B22 = quantize(B2 * quantize(v_fixed[t] / 2 ** 0,int_bits,frac_bits),int_bits,frac_bits)
        B33 = B3
        y = quantize(quantize(B33 + B22,int_bits,frac_bits) + B11,int_bits,frac_bits) # pA

        if x > 0:
            v_fixed.append(Vr)
            spiking_point_fixed.append(Vr)
            w_fixed.append(w_fixed[t] + b)
            spiking_time_fixed.append(t)
        elif thres > 20:
            v_fixed.append(Vr)
            spiking_point_fixed.append(Vr)
            w_fixed.append(w_fixed[t] + b)
            spiking_time_fixed.append(t)
        else:
            v_fixed.append(x)
            spiking_point_fixed.append(x)
            w_fixed.append(y)

        A011.append(A11)
        A022.append(A22)
        A033.append(A33)
        A044.append(A44)
        B011.append(B11)
        B022.append(B22)
        B033.append(B33)

    return spiking_point_fixed, spiking_time_fixed


if __name__ == '__main__':
    # 生成float point potential data并保存到文件
    spiking_point_float, spiking_time_float,w = GFloatV()
    # 生成fixed point potential data并保存到文件
    spiking_point_fixed, spiking_time_fixed = GFixedV()



    plt.figure
    plt.plot(spiking_point_float, 'black')
    plt.plot(spiking_point_fixed, 'r--')
    plt.xlabel('t (ms)')
    plt.title('v')
    plt.suptitle('Black:Software  Red:Hardware')
    plt.show()