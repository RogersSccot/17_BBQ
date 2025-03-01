#This is the original version of CORDIC algorithm
#Learn this version first to better understand the code

import numpy as np
########################  量化函数   ########################
def co_quantize(x, int_bits=6, frac_bits=19):
    sign = np.sign(x)
    qx = np.round(x * 2 ** frac_bits) / 2 ** frac_bits
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
    return np.cosh(np.arctanh(2 ** -x))

########################  查找表   ########################
RAM = np.zeros(100)
for i in range(100):
    RAM[i] = np.arctanh(2 ** (-(i + 1)))

########################  剩余角区间查找   ########################
def findProperAngle(mode, target, STEP):
    min = 100
    min_index = 100
    for i in range(STEP + 1):
        if mode == 1:
            if np.abs(np.abs(target) - RAM[i]) < min:
                min_index = i
                min = np.abs(np.abs(target) - RAM[i])
        else:
            if np.abs(np.abs(target) - 2 ** (-i)) < min:
                min_index = i
                min = np.abs(np.abs(target) - 2 ** (-i))
    if mode == 1:
        min_index = min_index + 1

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
    
    for i in range(STEP):
        if angleindex >= STEP:  #量化查找表的深度与 STEP 一致
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
            di      = - np.sign(x * y)

        angleindex = findProperAngle(mode, target, STEP)
        nx = co_quantize(x - u * di * y * (2 ** - angleindex))
        ny = co_quantize(y + di * x * (2 ** - angleindex))
        x = nx
        y = ny

        if mode == 1:
            z = co_quantize(z - di * RAM[angleindex-1])
            K = co_quantize(K * K_cal(angleindex))
        else:
            z = co_quantize(z - di * (2 ** -angleindex))

        cnt += 1

    x = co_quantize(x * K)
    y = co_quantize(y * K)
    return x, y, z, cnt


# mode = 1 输出 e ^ x     
# mode = 2 输出 x * y   
# mode = 3 输出 x / y 
def AR_CORDIC_quantize(mode, x, y, STEP = 16):
    if mode == 1:
        p = np.floor(x) #整数部分
        q = x - p       #小数部分
        coshq, sinhq, _, cnt = CORDIC_quantize(mode, 1, 0, q, STEP = STEP)
        y = coshq + sinhq                           # e**小数部分
        result = co_quantize(y * (np.e ** p))        # 整数部分直接用函数

    elif mode == 2:
        sign = np.sign(x * y)
        x = abs(x)
        y = abs(y)
        m, cntm = shift_2_range(x)
        n, cntn = shift_2_range(y)
        _, result, _, cnt = CORDIC_quantize(mode, m, 0, n, STEP = STEP)
        result = co_quantize(result * (2 ** (-cntm - cntn)) * sign)

    elif mode == 3:
        sign = np.sign(x * y)
        x = abs(x)
        y = abs(y)
        m, cntm = shift_2_range(y)
        n, cntn = shift_2_range(x)
        _,_,result,cnt = CORDIC_quantize(mode, m, n, 0, STEP = STEP)
        result = co_quantize(result * (2 ** (cntm - cntn)) * sign)

    else:
        result = 0
        cnt = 0

    return result

if __name__ == '__main__':
    res=   AR_CORDIC_quantize(1, 8.1, 0)
    print('result = ', res)
    # print('cnt = ',cnt)

