import numpy as np
import matplotlib.pyplot as plt
from RARCORDIC import *
from RARCORDIC_version4 import *

def quantize(x, int_bits=7, frac_bits=19):
    sign = np.sign(x)
    qx = np.floor(x * 2 ** frac_bits) / 2 ** frac_bits
    qx = np.clip(np.abs(qx), 0, 2 ** int_bits)
    qx = qx * sign
    return qx

######################  是否输出图像   ########################
plot_flag = True
# plot_flag = False
########################  初始赋值   ########################
I_p     = 20                # 输入电流
dt      = 2 ** (-7)          # 步进时长
T       = 50          # 仿真总时长
t       = np.arange(0, T, dt)
m_init  = 0.0529
n_init  = 0.3177
h_init  = 0.5961
v_init  = 0    # mV
########################  神经元生物常量   ########################
g_K_max     = 36  #18874368
g_Na_max    = 120  #62914560
g_L         = 0.3  #157286
E_K         = -72.14  #37822136
E_Na        = 55.12  #28898755
E_L         = -49.42  #25910313
V_rest      = 0  #
C_m         = 1.0  #524288
# g_K_max     = 5
# g_Na_max    = 50
# g_L         = 0.1
# E_K         = -100
# E_Na        = 50
# E_L         = -85
# V_rest      = 0
# C_m         = 1.0
##########################################################################################################################
####################################      HH 软件值 (调用CORDIC函数，不考虑量化误差)        ####################################
##########################################################################################################################
def AR_HH_float():
    cnt_div = 0
    cnt_exp = 0

    print("length =" , len(t))
    m = np.zeros(len(t))
    n = np.zeros(len(t))
    h = np.zeros(len(t))
    V_m = np.zeros(len(t))
    g_K = np.zeros(len(t))
    g_Na = np.zeros(len(t))

    V_m[1] = v_init
    m[1] = m_init
    n[1] = n_init
    h[1] = h_init

    alpha_m = np.zeros(len(t))
    beta_m = np.zeros(len(t))
    alpha_n = np.zeros(len(t))
    beta_n = np.zeros(len(t))
    alpha_h = np.zeros(len(t))
    beta_h = np.zeros(len(t))

    spiking_time = []
    spiking_point = []


    for i in range(1, len(t) - 1):

        exp0 = AR_CORDIC(1, -(V_m[i] * 0.1), 0)

        ################# alpha_m[i]  = 0.1 * ((V_m[i] + 35) / (1 - np.exp(- (0.1 * V_m[i] + 3.5))))
        alpha_m_son = (V_m[i] + 35) * 0.1
        alpha_m_mom = 1 - exp0 * (2**-5 - 2**-10 - 2**-14 - 2**-16)
        # alpha_m_mom = 1 - exp0 * (np.e ** -3.5)
        alpha_m[i] = AR_CORDIC(3,alpha_m_son,alpha_m_mom)

        ################# beta_m[i]   = 4 * np.exp(- (V_m[i] + 60) / 18)
        beta_m_exp = - (V_m[i] + 60) * (2**-5 + 2**-6 + 2**-7 + 2**-10 - 2**-13)
        # beta_m_exp = - (V_m[i] + 60) / 18
        beta_m[i] = AR_CORDIC(1,beta_m_exp,0) * 4

        ################## alpha_n[i]  = 0.01 * ((V_m[i] + 50) / (1 - np.exp(-(0.1 * V_m[i] + 5))))
        alpha_n_son = (V_m[i] + 50) * 0.01  # (2 ** -7 + 2 ** -9 + 2 ** -12)
        alpha_n_mom = 1 - exp0 * (np.e ** -5)  #(2**-8 + 2**-9 + 2**-10 - 2**-14 - 2**-15)
        alpha_n[i] = AR_CORDIC(3,alpha_n_son,alpha_n_mom)

        ################## beta_n[i]   = 0.125 * np.exp(- (V_m[i] + 60) / 80)
        beta_n_exp = - (V_m[i] + 60) * 0.0125
        beta_n[i] = AR_CORDIC(1, beta_n_exp, 0) * 0.125

        ################## alpha_h[i]  = 0.07 * np.exp(- (V_m[i] + 60) / 20)
        alpha_h_exp = - (V_m[i] + 60) * 0.05
        alpha_h[i] = AR_CORDIC(1,alpha_h_exp, 0) * (2**-4 + 2**-7 - 2**-12 - 2**-13 + 2**-14)
        # alpha_h[i] = AR_CORDIC(1, alpha_h_exp, 0) * 0.07


        ################## beta_h[i] = 1 / (np.exp(- (3 + 0.1 * V_m[i])) + 1)
        # beta_h_mom_exp = AR_CORDIC(1,-(V_m[i]*(2**-4 + 2**-5 + 2**-8 + 2**-9)),0)
        beta_h_mom = exp0 * (np.e ** -3) + 1
        beta_h[i] = AR_CORDIC(3,1,beta_h_mom)


        ################## g_Na[i]     = (m[i] ** 3) * g_Na_max * h[i]
        g_Na[i] = (m[i] ** 3) * g_Na_max * h[i]
        ################## g_K[i]      = (n[i] ** 4) * g_K_max
        g_K[i] = (n[i] ** 4) * g_K_max

        I_Na = g_Na[i] * (V_m[i] - E_Na)
        I_K = g_K[i] * (V_m[i] - E_K)
        I_L = g_L * (V_m[i] - E_L)

        I_ion = I_p - I_K - I_Na - I_L

        ########################  ``````````````````迭代公式``````````````````   ########################

        delta_v = I_ion / C_m * dt
        V_m[i + 1] = V_m[i] + delta_v
        m[i + 1] = m[i] + (alpha_m[i] - (alpha_m[i] + beta_m[i]) * m[i]) * dt
        n[i + 1] = n[i] + (alpha_n[i] - (alpha_n[i] + beta_n[i]) * n[i]) * dt
        h[i + 1] = h[i] + (alpha_h[i] - (alpha_h[i] + beta_h[i]) * h[i]) * dt

    for i in range(1, len(t) - 1):
        if V_m[i] > - V_rest:
            if (V_m[i] > V_m[i - 1]) and (V_m[i] > V_m[i + 1]):
                spiking_time.append(i)

    spiking_point = V_m + V_rest

    return spiking_point, spiking_time

##########################################################################################################################
####################################                    HH 理论值                      ####################################
##########################################################################################################################
def HH_theory():   
    alpha_m2    = np.zeros(len(t))
    beta_m2     = np.zeros(len(t))
    alpha_n2    = np.zeros(len(t))
    beta_n2     = np.zeros(len(t))
    alpha_h2    = np.zeros(len(t))
    beta_h2     = np.zeros(len(t))
    V_m2        = np.zeros(len(t))
    m2          = np.zeros(len(t))
    n2          = np.zeros(len(t))
    h2          = np.zeros(len(t))
    g_K2        = np.zeros(len(t))
    g_Na2       = np.zeros(len(t))
    V_m2[0]     = v_init
    m2[0]       = m_init
    n2[0]       = n_init
    h2[0]       = h_init
    spiking_time = []
    spiking_point = []



    for i in range(len(t) - 1):
        alpha_m2[i]     = 0.1   * ((V_m2[i] + 35) / (1 - np.exp(- (0.1 * V_m2[i] + 3.5))))
        beta_m2[i]      = 4     * np.exp(- (V_m2[i] + 60) / 18)
        alpha_n2[i]     = 0.01  * ((V_m2[i] + 50) / (1 - np.exp(-(0.1 * V_m2[i] + 5))))
        beta_n2[i]      = 0.125 * np.exp(- (V_m2[i] + 60) / 80)
        alpha_h2[i]     = 0.07  * np.exp(- (V_m2[i] + 60) / 20)
        beta_h2[i]      = 1    / (np.exp(- (3 + 0.1 * V_m2[i])) + 1)

        g_Na2[i]    = m2[i] ** 3 * g_Na_max * h2[i]
        g_K2[i]     = n2[i] ** 4 * g_K_max
        I_Na2       = g_Na2[i] * (E_Na - V_m2[i])
        I_K2        = g_K2[i]  * (E_K - V_m2[i])
        I_L2        = g_L      * (E_L - V_m2[i])
        I_ion       = I_p + I_K2 + I_Na2 + I_L2


        V_m2[i + 1] = V_m2[i] + I_ion / C_m * dt
        m2[i + 1]   = m2[i] + (alpha_m2[i] * (1 - m2[i]) - beta_m2[i] * m2[i]) * dt
        n2[i + 1]   = n2[i] + (alpha_n2[i] * (1 - n2[i]) - beta_n2[i] * n2[i]) * dt
        h2[i + 1]   = h2[i] + (alpha_h2[i] * (1 - h2[i]) - beta_h2[i] * h2[i]) * dt

    

    for i in range(1,len(t) - 1):
        if V_m2[i] >  V_rest:
            if (V_m2[i] > V_m2[i - 1]) and (V_m2[i] > V_m2[i + 1]):
                spiking_time.append(i)

    spiking_point = V_m2 + V_rest

    return spiking_point, spiking_time


##########################################################################################################################
####################################                    HH 硬件值                      ####################################
##########################################################################################################################
def AR_HH_fixed():

    m = np.zeros(len(t))
    n = np.zeros(len(t))
    h = np.zeros(len(t))
    V_m = np.zeros(len(t))
    g_K = np.zeros(len(t))
    g_Na = np.zeros(len(t))
    V_m[0] = v_init
    m[0] = m_init
    n[0] = n_init
    h[0] = h_init

    v_01 = np.zeros(len(t))
    alpha_m = np.zeros(len(t))
    alpha_m_son = np.zeros(len(t))  # son：分子
    alpha_m_mom = np.zeros(len(t))  # mom：分母
    beta_m = np.zeros(len(t))
    beta_m_exp = np.zeros(len(t))
    alpha_n = np.zeros(len(t))
    alpha_n_son = np.zeros(len(t))
    alpha_n_mom = np.zeros(len(t))
    beta_n = np.zeros(len(t))
    beta_n_exp = np.zeros(len(t))
    alpha_h = np.zeros(len(t))
    alpha_h_exp = np.zeros(len(t))
    beta_h = np.zeros(len(t))
    beta_h_son = np.zeros(len(t))
    beta_h_mom = np.zeros(len(t))
    I_Na = np.zeros(len(t))
    I_K = np.zeros(len(t))
    I_L = np.zeros(len(t))
    delta_v = np.zeros(len(t))
    I_ion = np.zeros(len(t))

    spiking_time = []
    spiking_point = []


    for i in range(len(t) - 1):
        exp0,cnt = AR_CORDIC_quantize(1, - quantize(V_m[i] * 0.1 + 3), 0)
        v_01[i] = - quantize(V_m[i] * 0.1)

        ################# alpha_m[i]  = 0.1 * ((V_m[i] + 35) / (1 - np.exp(- (0.1 * V_m[i] + 3.5))))
        alpha_m_son[i] = quantize(quantize(V_m[i] + 35) * (52429/2**19))
        alpha_m_mom[i] = quantize(1 - quantize(exp0 * (317997/2**19)))#e^-0.5
        alpha_m[i],cnt = AR_CORDIC_quantize(3, alpha_m_son[i], alpha_m_mom[i])
        # alpha_m[i] =  alpha_m_son / alpha_m_mom

        ################# beta_m[i]   = 4 * np.exp(- (V_m[i] + 60) / 18)
        beta_m_exp[i] = - quantize(quantize(V_m[i] + 60) * (29127/2**19))
        beta_m0,cnt = AR_CORDIC_quantize(1, beta_m_exp[i], 0)
        beta_m[i] = quantize(beta_m0 * 4)

        ################## alpha_n[i]  = 0.01 * ((V_m[i] + 50) / (1 - np.exp(-(0.1 * V_m[i] + 5))))
        alpha_n_son[i] = quantize(quantize(V_m[i] + 50) * (5243/2**19)) # (2 ** -7 + 2 ** -9 + 2 ** -12)
        alpha_n_mom[i] = quantize(1 - quantize(exp0 * (70955/2**19)))  # (2**-8 + 2**-9 + 2**-10 - 2**-14 - 2**-15)
        alpha_n[i],cnt = AR_CORDIC_quantize(3,alpha_n_son[i] ,alpha_n_mom[i])
        # alpha_n[i] = alpha_n_son / alpha_n_mom

        ################## beta_n[i]   = 0.125 * np.exp(- (V_m[i] + 60) / 80)
        beta_n_exp[i] = - quantize(quantize(V_m[i] + 60) * (6554/2**19))
        beta_n0,cnt = AR_CORDIC_quantize(1, beta_n_exp[i], 0)
        beta_n[i] = quantize(beta_n0 * 0.125)

        ################## alpha_h[i]  = 0.07 * np.exp(- (V_m[i] + 60) / 20)
        alpha_h_exp[i] = - quantize(quantize(V_m[i] + 60) * (26215/2**19))
        alpha_h0,cnt = AR_CORDIC_quantize(1, alpha_h_exp[i], 0)
        alpha_h[i] = quantize(alpha_h0 * (36700/2**19))
        # alpha_h[i] = quantize(AR_CORDIC_quantize(1, alpha_h_exp[i], 0) * (2 ** -4 + 2 ** -7 - 2 ** -12 - 2 ** -13 + 2 ** -14))

        ################## beta_h[i] = 1 / (np.exp(- (3 + 0.1 * V_m[i])) + 1)
        # beta_h_mom_exp = AR_CORDIC(1,-(V_m[i]*(2**-4 + 2**-5 + 2**-8 + 2**-9)),0)
        beta_h_son[i] = 1
        beta_h_mom[i] = quantize(exp0 + 1)  # (2**-5 + 2**-6 + 2**-8 - 2**-10 + 2**-16)
        beta_h[i],cnt = AR_CORDIC_quantize(3, beta_h_son[i], beta_h_mom[i])
        # beta_h[i] = 1 / beta_h_mom

        ################## g_Na[i]     = (m[i] ** 3) * g_Na_max * h[i]
        g_Na[i] = quantize(quantize(quantize(quantize(m[i] * m[i]) * m[i]) * h[i] ) * g_Na_max)
        ################## g_K[i]      = (n[i] ** 4) * g_K_max
        n2 = quantize(n[i] * n[i])
        g_K[i] = quantize(quantize(n2 * n2) * g_K_max)

        I_Na[i] = quantize(g_Na[i] * quantize(quantize(E_Na-V_m[i])*dt))
        I_K[i] = quantize(g_K[i] * quantize(quantize(E_K- V_m[i])*dt))
        I_L[i] = quantize(g_L * quantize(quantize(E_L - V_m[i])*dt))

        I_ion[i] = quantize(quantize(quantize(I_K[i] + I_L[i]) + I_Na[i]) +  81920/2**19)

        ########################  ``````````````````迭代公式``````````````````   ########################

        delta_v[i] = quantize(I_ion[i] / C_m)
        V_m[i + 1] = quantize(V_m[i] + delta_v[i])
        m[i + 1] = m[i] + quantize(quantize(alpha_m[i] - quantize(quantize((alpha_m[i] + beta_m[i]) * m[i]))) * dt)
        n[i + 1] = n[i] + quantize(quantize(alpha_n[i] - quantize(quantize((alpha_n[i] + beta_n[i]) * n[i]))) * dt)
        h[i + 1] = h[i] + quantize(quantize(alpha_h[i] - quantize(quantize((alpha_h[i] + beta_h[i]) * h[i]))) * dt)

    for i in range(1, len(t) - 1):
        if V_m[i] >  V_rest:
            if (V_m[i] > V_m[i - 1]) and (V_m[i] > V_m[i + 1]):
                spiking_time.append(i)

    spiking_point = V_m + V_rest
    
    return spiking_point, spiking_time,m,n,h,alpha_h,beta_h
##########################################################################################################################
####################################                    画      图                      ####################################
##########################################################################################################################
if __name__ == '__main__':

    spiking_point_theory, spiking_time_theory = HH_theory()
    spiking_point_float, spiking_time_float = AR_HH_float()
    spiking_point_fixed, spiking_time_fixed,m,n,h,alpha_h,beta_h= AR_HH_fixed()


    if plot_flag is True:
        plt.figure()

        plt.plot(spiking_point_theory,   'g-')
        plt.plot(spiking_point_float ,   'blue')
        plt.plot(spiking_point_fixed ,   'r--')
        plt.xlabel('t (ms)')
        plt.title(f'I = {I_p} uA/cm^2  Blue:Software  Red:Hardware  Green:Theory')
        plt.show()
