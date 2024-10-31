# noinspection LanguageDetectionInspection
import numpy as np
import matplotlib.pyplot as plt
#This is a version of CORDIC algorithm that do not consider hardware bitwidth
#All the variables are "float" type

########################  除法提前处理函数   ########################
def shift_2_range(x):
    out = 0
    cnt = 0
    if(x<0.5):
        for i in range(50):#此左右位移数目可以依照定点数位数改变
            if(( x * 2 ** i ) >= 0.5):
                x = x * 2 ** i
                out = x
                cnt = i
                break
    elif x>1:
        for i in range(50): #此左右位移数目可以依照定点数位数改变
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
    return np.cosh(np.arctanh(2 ** (-x)))

########################  查找表   ########################
RAM = np.zeros(100)
for i in range(100):
    RAM[i] = np.arctanh(2 ** (-(i + 1)))

########################  剩余角区间查找   ########################
def findProperAngle(mode, target, STEP):
    min = 1000
    min_index = 1
    for i in range(STEP+1):
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
########################  量化函数   ########################
def quantize(x,frac_bit):
    if x >= 0:
        x = x * (2 ** frac_bit)
        x = np.floor(x)
    else:
        x = x * (2 ** frac_bit)
        x = np.ceil(x)
    x = x / (2 ** frac_bit)
    return x

# mode = 1 指数
# mode = 2 乘法
# mode = 3 除法
def CORDIC(mode, x, y, z, STEP):
    cnt = 0
    # z = X
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
        # print("angleindex = ",angleindex)
        # print("di = ",di)
        nx = x - u * di * y * (2 ** - angleindex)
        ny = y + di * x * (2 ** - angleindex)
        x = nx
        y = ny

        if mode == 1:
            z = z - di * RAM[angleindex-1]
            # z = z - di * RAM[angleindex]
            K = K * K_cal(angleindex)  #模长修正
        else:
            z = z - di * (2 ** (-angleindex))

        cnt += 1

    x = x * K
    y = y * K
    # print("y = ",y)
    return x, y, z, cnt

# mode = 1 输出 e ^ x
# mode = 2 输出 x * y
# mode = 3 输出 x / y
STEP = 16
def AR_CORDIC(mode, x, y):
    if mode == 1:
        p = np.floor(x) #整数部分
        q = x - p       #小数部分
        coshq, sinhq,_,cnt = CORDIC(mode, 1, 0, q, STEP = STEP)
        y = coshq + sinhq               # e**小数部分
        result = y * (np.e ** p)        # 整数部分直接用函数

    elif mode == 2:
        sign = np.sign(x * y)
        x = abs(x)
        y = abs(y)
        m, cntm = shift_2_range(x)
        n, cntn = shift_2_range(y)
        _,result,_,cnt = CORDIC(mode, m, 0, n, STEP = STEP)
        result = result * 2 ** (-cntm - cntn) * sign
        # print("n = ",n)
        # print("m = ",m)


    elif mode == 3:
        sign = np.sign(x * y)
        x = abs(x)
        y = abs(y)
        m, cntm = shift_2_range(y)
        n, cntn = shift_2_range(x)
        _,_,result,cnt = CORDIC(mode, m, n, 0, STEP = STEP)
        result = result * 2 ** (cntm - cntn) * sign


    else:
        result = 0
        cnt = 0

    return result

def accuracy_fun():
    theory = []
    cordic = []
    acc = 0
    cnt = 0
    for i in range (1,100000):
        for j in range(1,600000):
            son =  0.0001 * i
            mom =  0.0001 * j
            test, cnt0 = AR_CORDIC(3, son, mom)
            cordic.append(test)
            ans = son / mom
            theory.append(ans)
            accuracy = test - ans
            ratio = abs(test - ans) / ans
            acc = acc + ratio
            cnt = cnt + 1
    ave = acc / cnt
    print("average error =",ave)


if __name__ == '__main__':
    # accuracy_fun()
    for i in range(16):
        print("i=", i)
        print("K[i] = ",K_cal(i))