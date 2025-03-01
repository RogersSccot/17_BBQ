import math
from shifter import *
# This is the final version of CORDIC algorithm
# It is modified on the basis of the original version
# in order to simulate all the details of hardware implementation
import numpy as np
########################  量化函数   ########################
def co_quantize_in(x, int_bits=4, frac_bits=16):
    sign = np.sign(x)
    qx = np.floor(x * 2 ** frac_bits) / 2 ** frac_bits
    qx = np.clip(np.abs(qx), 0, 2 ** int_bits)
    qx = qx * sign
    return qx
def co_quantize_out(x, int_bits=6, frac_bits=16):
    sign = np.sign(x)
    qx = np.floor(x * 2 ** frac_bits) / 2 ** frac_bits
    qx = np.clip(np.abs(qx), 0, 2 ** int_bits)
    qx = qx * sign
    return qx
########################  乘法、除法提前处理函数   ########################
def shift_2_range(x):
    out = 0
    cnt = 0
    if(x<0.5):
        for i in range(1,50):           #此左右位移数目可以依照定点数位数改变
            if(( x * 2 ** i ) >= 0.5):
                x = x * 2 ** i
                out = x
                cnt = i
                break
    elif x>1:
        for i in range(1,50):           #此左右位移数目可以依照定点数位数改变
            if(( x / 2 ** i ) <= 1):
                x = x / 2 ** i
                out = x
                cnt = -i
                break
    else:
        out = x
        cnt = 0
    return out,cnt

########################  模长补偿因子   ########################
def K_cal(x):
    return co_quantize_in(np.cosh(np.arctanh(2 ** -x)))

########################  查找表   ########################
RAM = np.zeros(100)
for i in range(1,100):
    RAM[i] = co_quantize_in(np.arctanh(2 ** (-i)))
RAM[17] = 1 / 2**16
RAM[15] = 1 / 2**16
########################  剩余角区间查找   ########################
def findProperAngle(mode, target, STEP,x):
    if ((mode == 1) & (target >= 26369 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-2)))):
        min_index = 1
    elif ((mode == 1) & (target >= 12487 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-3)))):
        min_index = 2
    elif ((mode == 1) & (target >= 6168 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-4)))):
        min_index = 3
    elif ((mode == 1) & (target >= 3075 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-5)))):
        min_index = 4
    elif ((mode == 1) & (target >= 1537 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-6)))):
        min_index = 5
    elif ((mode == 1) & (target >= 768 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-7)))):
        min_index = 6
    elif ((mode == 1) & (target >= 384 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-8)))):
        min_index = 7
    elif ((mode == 1) & (target >= 192 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-9)))):
        min_index = 8
    elif ((mode == 1) & (target >= 96 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-10)))):
        min_index = 9
    elif ((mode == 1) & (target >= 48 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-11)))):
        min_index = 10
    elif ((mode == 1) & (target >= 24 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-12)))):
        min_index = 11
    elif ((mode == 1) & (target >= 12 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-13)))):
        min_index = 12
    elif ((mode == 1) & (target >= 6 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-14)))):
        min_index = 13
    elif ((mode == 1) & (target >= 3 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-15)))):
        min_index = 14
    elif ((mode == 1) & (target >= 1 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-16)))):
        min_index = 15
    elif ((mode == 1) & (target >= 0 / 2**16))|((mode == 3) & (target >= co_quantize_in(1.5 * x * 2 **(-17)))):
        min_index = 16
    else:
        min_index = 17

    return min_index



########################################################################################################################
####################################                    CORDIC                      ####################################
########################################################################################################################
# mode = 1 指数
# mode = 2 乘法
# mode = 3 除法
def CORDIC_quantize(mode, x, y, z, STEP):
    cnt = 0
    K = 1
    angleindex = 0              #角度索引值
    index_mem = 0
    # print(" ")
    for i in range(STEP):
        if angleindex >= STEP :
             #量化查找表的深度与 STEP 一致
            break

        if mode == 1:           #指数
            target  = z
            u       = -1
            di      = np.sign(z)
        elif mode == 2:         #乘法
            target  = z
            u       = 0
            di      = np.sign(z)
        else:                   #除法
            target  = y
            u       = 0
            di      = - np.sign(x*y)

        angleindex = findProperAngle(mode, abs(target), STEP, x)

        if angleindex == 1:
            index_mem = index_mem + 1
        elif angleindex == 2:
            index_mem = index_mem + 4
        elif angleindex == 3:
            index_mem = index_mem + 8
        elif angleindex == 4:
            index_mem = index_mem + 16
        else :
            index_mem = index_mem

        nx = co_quantize_in(x - u * di * co_quantize_in(y * (2 ** - angleindex)))
        ny = co_quantize_in(y + di * co_quantize_in(x * (2 ** - angleindex)))
        x = nx
        y = ny

        if mode == 1:
            z = co_quantize_in(z - di * co_quantize_in(RAM[angleindex]))

        else:
            z = co_quantize_in(z - di * co_quantize_in((2 ** -angleindex)))

        cnt += 1
    # use multiplier directly
    # if mode == 1:
    #     if index_mem == 13:
    #         x = x * 1.201983356
    #         y = y * 1.201983356
    #     elif index_mem == 5:
    #         x = x * 1.19255732
    #         y = y * 1.19255732
    #     elif index_mem == 9:
    #         x = x * 1.163820363
    #         y = y * 1.163820363
    #     elif index_mem == 12:
    #         x = x * 1.040954373
    #         y = y * 1.040954373
    #     elif index_mem == 1:
    #         x = x * 1.154693604
    #         y = y * 1.154693604
    #     elif index_mem == 4:
    #         x = x * 1.032791138
    #         y = y * 1.032791138
    #     elif index_mem == 8:
    #         x = x * 1.007904053
    #         y = y * 1.007904053
    #     elif index_mem == 10:
    #         x = x * 1.343855928
    #         y = y * 1.343855928
    #     elif index_mem == 16:
    #         x = x * 1.001953125
    #         y = y * 1.001953125
    #     elif index_mem == 17:
    #         x = x * 1.156948864
    #         y = y * 1.156948864
    #     elif index_mem == 20:
    #         x = x * 1.034808308
    #         y = y * 1.034808308
    #     elif index_mem == 21:
    #         x = x * 1.194886534
    #         y = y * 1.194886534
    #     elif index_mem == 24:
    #         x = x * 1.009872615
    #         y = y * 1.009872615
    #     elif index_mem == 25:
    #         x = x * 1.166093449
    #         y = y * 1.166093449
    #     else:
    #         x = x
    #         y = y
    # else:
    #     x = x
    #     y = y


    # use shifter and adder to implement multiplication
    # if mode == 1:
    #     if index_mem == 1:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-3 + 2**-5 - 2**-9 + 2**-12 + 2**-13 + 2**-15))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-3 + 2**-5 - 2**-9 + 2**-12 + 2**-13 + 2**-15))
    #     elif index_mem == 4:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-5 + 2**-10 + 2**-11 + 2**-14 + 2**-16))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-5 + 2**-10 + 2**-11 + 2**-14 + 2**-16))
    #     elif index_mem == 5:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-3 + 2**-4 + 2**-8 + 2**-10 + 2**-13 + 2**-15 + 2**-16))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-3 + 2**-4 + 2**-8 + 2**-10 + 2**-13 + 2**-15 + 2**-16))
    #     elif index_mem == 8:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-7 + 2**-14 + 2**-15))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-7 + 2**-14 + 2**-15))
    #     elif index_mem == 9:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-3 + 2**-5 + 2**-7 - 2**-12))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-3 + 2**-5 + 2**-7 - 2**-12))
    #     elif index_mem == 10:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-2 + 2**-4 + 2**-5 + 2**-13))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-2 + 2**-4 + 2**-5 + 2**-13))
    #     elif index_mem == 12:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-5 + 2**-7 + 2**-9 - 2**-14))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-5 + 2**-7 + 2**-9 - 2**-14))
    #     elif index_mem == 13:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-3 + 2**-4 + 2**-6 - 2**-9 + 2**-11 + 2**-12))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-3 + 2**-4 + 2**-6 - 2**-9 + 2**-11 + 2**-12))
    #     elif index_mem == 16:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-9))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-9))
    #     elif index_mem == 17:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-3 + 2**-5 + 2**-11 + 2**-12 - 2**-14))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-3 + 2**-5 + 2**-11 + 2**-12 - 2**-14))
    #     elif index_mem == 20:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-5 + 2**-8 - 2**-11 + 2**-13 + 2**-16))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-5 + 2**-8 - 2**-11 + 2**-13 + 2**-16))
    #     elif index_mem == 21:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-3 + 2**-4 + 2**-7 - 2**-11 + 2**-14))
    #         y = co_quantize_in(y * (1 + 2**-3 + 2**-4 + 2**-7 - 2**-11 + 2**-14))
    #     elif index_mem == 24:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-7 + 2**-9 + 2**-13 - 2**-16))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-7 + 2**-9 + 2**-13 - 2**-16))
    #     elif index_mem == 25:
    #         x = co_quantize_in(x * co_quantize_in(1 + 2**-3 + 2**-5 + 2**-7 + 2**-9 + 2**-14 + 2**-16))
    #         y = co_quantize_in(y * co_quantize_in(1 + 2**-3 + 2**-5 + 2**-7 + 2**-9 + 2**-14 + 2**-16))
    #     else:
    #         x = x
    #         y = y
    # else:
    #     x = x
    #     y = y

    # take rounding errors into consideration in "shifter and adder" operation
    if mode == 1:
        if index_mem == 1:
            x = co_quantize_in(
                x + shifter_4_16(x, 3) + shifter_4_16(x, 5) - shifter_4_16(x, 9) + shifter_4_16(x, 12) + shifter_4_16(x,
                                                                                                                      13) + shifter_4_16(
                    x, 15))
            y = co_quantize_in(
                y + shifter_4_16(y, 3) + shifter_4_16(y, 5) - shifter_4_16(y, 9) + shifter_4_16(y, 12) + shifter_4_16(y,
                                                                                                                      13) + shifter_4_16(
                    y, 15))
        elif index_mem == 4:
            x = co_quantize_in(
                x + shifter_4_16(x, 5) + shifter_4_16(x, 10) + shifter_4_16(x, 11) + shifter_4_16(x, 14) + shifter_4_16(
                    x, 16))
            y = co_quantize_in(
                y + shifter_4_16(y, 5) + shifter_4_16(y, 10) + shifter_4_16(y, 11) + shifter_4_16(y, 14) + shifter_4_16(
                    y, 16))
        elif index_mem == 5:
            x = co_quantize_in(
                x + shifter_4_16(x, 3) + shifter_4_16(x, 4) + shifter_4_16(x, 8) + shifter_4_16(x, 10) + shifter_4_16(x,
                                                                                                                      13) + shifter_4_16(
                    x, 15) + shifter_4_16(x, 16))
            y = co_quantize_in(
                y + shifter_4_16(y, 3) + shifter_4_16(y, 4) + shifter_4_16(y, 8) + shifter_4_16(y, 10) + shifter_4_16(y,
                                                                                                                      13) + shifter_4_16(
                    y, 15) + shifter_4_16(y, 16))
        elif index_mem == 8:
            x = co_quantize_in(x + shifter_4_16(x, 7) + shifter_4_16(x, 14) + shifter_4_16(x, 15))
            y = co_quantize_in(y + shifter_4_16(y, 7) + shifter_4_16(y, 14) + shifter_4_16(y, 15))
        elif index_mem == 9:
            x = co_quantize_in(x + shifter_4_16(x, 3) + shifter_4_16(x, 5) + shifter_4_16(x, 7) - shifter_4_16(x, 12))
            y = co_quantize_in(y + shifter_4_16(y, 3) + shifter_4_16(y, 5) + shifter_4_16(y, 7) - shifter_4_16(y, 12))
        elif index_mem == 10:
            x = co_quantize_in(x + shifter_4_16(x, 2) + shifter_4_16(x, 4) + shifter_4_16(x, 5) + shifter_4_16(x, 13))
            y = co_quantize_in(y + shifter_4_16(y, 2) + shifter_4_16(y, 4) + shifter_4_16(y, 5) + shifter_4_16(y, 13))
        elif index_mem == 12:
            x = co_quantize_in(x + shifter_4_16(x, 5) + shifter_4_16(x, 7) + shifter_4_16(x, 9) - shifter_4_16(x, 14))
            y = co_quantize_in(y + shifter_4_16(y, 5) + shifter_4_16(y, 7) + shifter_4_16(y, 9) - shifter_4_16(y, 14))
        elif index_mem == 13:
            x = co_quantize_in(
                x + shifter_4_16(x, 3) + shifter_4_16(x, 4) + shifter_4_16(x, 6) - shifter_4_16(x, 9) + shifter_4_16(x,
                                                                                                                     11) + shifter_4_16(
                    x, 12))
            y = co_quantize_in(
                y + shifter_4_16(y, 3) + shifter_4_16(y, 4) + shifter_4_16(y, 6) - shifter_4_16(y, 9) + shifter_4_16(y,
                                                                                                                     11) + shifter_4_16(
                    y, 12))
        elif index_mem == 16:
            x = co_quantize_in(x + shifter_4_16(x, 9))
            y = co_quantize_in(y + shifter_4_16(y, 9))
        elif index_mem == 17:
            x = co_quantize_in(
                x + shifter_4_16(x, 3) + shifter_4_16(x, 5) + shifter_4_16(x, 11) + shifter_4_16(x, 12) - shifter_4_16(
                    x, 14))
            y = co_quantize_in(
                y + shifter_4_16(y, 3) + shifter_4_16(y, 5) + shifter_4_16(y, 11) + shifter_4_16(y, 12) - shifter_4_16(
                    y, 14))
        elif index_mem == 20:
            x = co_quantize_in(
                x + shifter_4_16(x, 5) + shifter_4_16(x, 8) - shifter_4_16(x, 11) + shifter_4_16(x, 13) + shifter_4_16(x, 16))
            y = co_quantize_in(y + shifter_4_16(y, 5) + shifter_4_16(y, 8) - shifter_4_16(y, 11) + shifter_4_16(y, 13) + shifter_4_16(y, 16))
        elif index_mem == 21:
            x = co_quantize_in(
                x + shifter_4_16(x, 3) + shifter_4_16(x, 4) + shifter_4_16(x, 7) - shifter_4_16(x, 11) + shifter_4_16(x,14))
            y = co_quantize_in(
                y + shifter_4_16(y, 3) + shifter_4_16(y, 4) + shifter_4_16(y, 7) - shifter_4_16(y, 11) + shifter_4_16(y,14))
        elif index_mem == 24:
            x = co_quantize_in(x + shifter_4_16(x, 7) + shifter_4_16(x, 9) + shifter_4_16(x, 13) - shifter_4_16(x, 16))
            y = co_quantize_in(y + shifter_4_16(y, 7) + shifter_4_16(y, 9) + shifter_4_16(y, 13) - shifter_4_16(y, 16))
        elif index_mem == 25:
            x = co_quantize_in(x + shifter_4_16(x, 3) + shifter_4_16(x, 5) + shifter_4_16(x, 7) + shifter_4_16(x, 9) + shifter_4_16(x,14) + shifter_4_16(x, 16))
            y = co_quantize_in(y + shifter_4_16(y, 3) + shifter_4_16(y, 5) + shifter_4_16(y, 7) + shifter_4_16(y, 9) + shifter_4_16(y,14) + shifter_4_16(y, 16))
        else:
            x = x
            y = y
    else:
        x = x
        y = y
    return x, y, z, cnt


# mode = 1 输出 e ^ x
# mode = 2 输出 x * y
# mode = 3 输出 x / y
def AR_CORDIC_quantize(mode, x, y, STEP = 16):
    if mode == 1:
        p = int(x) #整数部分
        q = co_quantize_in(x - p)       #小数部分
        coshq, sinhq, _, cnt = CORDIC_quantize(mode, 1, 0, q, STEP = STEP)
        result = coshq + sinhq                            # e**小数部分
        while (p != 0):
            if (p > 0):
                # result = quantize_out(result * 2.718281828)
                result = co_quantize_out(shifter_6_16(result,-1) + shifter_6_16(result,1) + shifter_6_16(result,3) + shifter_6_16(result,4) + shifter_6_16(result,5) - shifter_6_16(result,11) + shifter_6_16(result,16) + shifter_6_16(result,18) + shifter_6_16(result,20))
                p = p - 1
            elif (p < 0):
                # result = quantize_out(result * 0.3678794412)
                result = co_quantize_out(shifter_6_16(result,2) + shifter_6_16(result,3) - shifter_6_16(result,7) + shifter_6_16(result,11) + shifter_6_16(result,13) + shifter_6_16(result,16) + shifter_6_16(result,18) + shifter_6_16(result,20) + shifter_6_16(result,21))
                p = p + 1
            else:
                break

    elif mode == 2:
        sign = np.sign(x * y)
        x = abs(x)
        y = abs(y)
        m, cntm = shift_2_range(x)
        n, cntn = shift_2_range(y)
        _, result, _, cnt = CORDIC_quantize(mode, m, 0, n, STEP = STEP)
        result = co_quantize_out(result * (2 ** (-cntm - cntn)) * sign)

    elif mode == 3:
        sign = np.sign(x * y)
        x = co_quantize_in(abs(x) / 4)
        y = co_quantize_in(abs(y) / 4)
        _,_,result,cnt = CORDIC_quantize(mode, y, x, 0, STEP = STEP)
        result = co_quantize_out(result * sign)

    else:
        result = 0
        cnt = 0

    return result,cnt

if __name__ == '__main__':
    CORDIC_in = -714850
    result=AR_CORDIC_quantize(1, CORDIC_in/(2**16) ,0)
    print('result =', result * (2**16))




