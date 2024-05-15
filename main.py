# This is a sample Python script.
import math
import numpy as np
from datetime import datetime
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

# l = 1, h = 0.1 + 0.05
# l = 20, h = 0.5

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
@jit(parallel=True, nopython=False)
def count_Unmk(l: int, h, n: int, m: int, k: int, u, T: int, tau: float):
    sqr = math.sqrt(8 / (l ** 3)) * h ** 3

    Unmk = [0.] * (int(T / tau) + 1)
    t = 0
    while t <= T:
        i = 0
        while i <= l:
            j = 0
            while j <= l:
                q = 0
                while q <= l:
                    Unmk[int(t / tau)] += u[int(i / h), int(j / h), int(q / h), int(t / tau)] * sqr * math.cos(
                        math.pi * n * i / l) * math.cos(math.pi * m * j / l) * math.cos(math.pi * k * q / l)
                    q += h
                j += h
            i += h
        t += tau
    # print("i found unmk")
    return np.array(Unmk, dtype="float64")


@jit(parallel=True)
def count_Unmk_integral(delta2: float, T: int, Unmk: list, tau: float):
    # print("start unmk intrgral")
    ret_int = 0
    for t in range(0, T + 1):
        ret_int += math.exp(delta2 * (T * tau - t * tau)) * Unmk[t] * tau  # проверить
    # print("finish unmk intrgral")
    return ret_int


@jit(parallel=True, nopython=False)
def count_f(l: int, h: float, u, T: int, tau: float):
    N = 3
    sqr2 = 8 / (l ** 3)

    # Ensure that division results in integers
    T_div_tau = int(T / tau)
    l_div_h = int(l / h)

    # Create the array using the computed integer values
    my_sum = np.zeros((T_div_tau + 1, l_div_h + 1, l_div_h + 1, l_div_h + 1), dtype="float64")

    for t in range(0, int(T / tau) + 1):
        print("t", t)
        for x in range(0, int(l / h) + 1):
            for y in range(0, int(l / h) + 1):
                for z in range(0, int(l / h) + 1):
                    for n in range(1, N):
                        for m in range(1, N):
                            for k in range(1, N):
                                # print(x, y, z, n, m, k)
                                delta2 = (math.pi * n / l) ** 2 + (math.pi * m / l) ** 2 + (math.pi * k / l) ** 2
                                # print(delta2)
                                # print("h1")
                                integral = count_Unmk_integral(delta2, t, count_Unmk(l, h, n, m, k, u, T, tau), tau)
                                # print(integral)
                                # print("h2")
                                my_sum[x][y][z][t] += (
                                            math.cos(math.pi * n * x * h / l) * math.cos(math.pi * m * y * h / l)
                                            * math.cos(math.pi * k * z * h / l) * integral)
                                # print("h3")
                    my_sum[x][y][z][t] = my_sum[x][y][z][t] * sqr2

    return my_sum


@jit(parallel=True, nopython=False)
def count_psi(l, h, b, f, T, tau, a):
    shape = (int(T / tau) + 1, int(l / h) + 1, int(l / h) + 1, int(l / h) + 1)
    psi = np.zeros(shape, dtype="float64")

    for i in range(int(l / h) + 1):
        for j in range(int(l / h) + 1):
            for k in range(int(l / h) + 1):
                psi[i][j][k][0] = f[i][j][k][int(T / tau)] - b[i][j][k]
    # print("psi",psi)
    a_h = (a * a) / (h * h)

    for t in range(int(T / tau)):
        for i in range(int(l / h) + 1):
            for j in range(int(l / h) + 1):
                for k in range(int(l / h) + 1):
                    i_minus = 0
                    j_minus = 0
                    k_minus = 0
                    i_plus = 0
                    j_plus = 0
                    k_plus = 0
                    if i != 0:
                        i_minus = psi[i - 1][j][k][t]
                    if j != 0:
                        j_minus = psi[i][j - 1][k][t]
                    if k != 0:
                        k_minus = psi[i][j][k - 1][t]
                    if i != int(l / h):
                        i_plus = psi[i + 1][j][k][t]
                    if j != int(l / h):
                        j_plus = psi[i][j + 1][k][t]
                    if k != int(l / h):
                        k_plus = psi[i][j][k + 1][t]
                    psi[i][j][k][t + 1] = tau * (psi[i][j][k][t] / tau +
                                                 (6 * a_h) * psi[i][j][k][t] -
                                                 a_h * (i_plus + i_minus +
                                                        j_plus + j_minus +
                                                        k_plus + k_minus))
    # print("psi", psi)
    return psi


@jit(parallel=True, nopython=False)
def count_q(l, h, qk, psi, ak):
    qk1 = np.array(
        [[[[0 for _ in range(T_tau + 1)] for _ in range(l_h + 1)] for _ in range(l_h + 1)] for _ in range(l_h + 1)],
        dtype="float64")

    for t in range(int(T / tau) + 1):
        for i in range(int(l / h) + 1):
            for j in range(int(l / h) + 1):
                for k in range(int(l / h) + 1):
                    qk1[i][j][k][t] = qk[i][j][k][t] - 2 * ak * psi[i][j][k][int(T / tau) - t]
    # print("I found q")
    return qk1


def compare_control(fk, fk1, T, tau, l, h):
    sum = 0
    sum_1 = 0
    znamen = 0
    sum_kv = 0
    sum_kv1 = 0
    for i in range(int(l / h) + 1):
        for j in range(int(l / h) + 1):
            for k in range(int(l / h) + 1):
                sum_kv += fk[i][j][k] * fk[i][j][k]
                sum_kv1 += fk1[i][j][k] * fk1[i][j][k]
                sum += abs(fk[i][j][k])
                sum_1 += abs(fk1[i][j][k])
                znamen += 1
    # print('sum', sum_kv)
    # print('sum_1', sum_kv1)
    # if sum_kv1 < sum_kv:
    #     return True
    print('sum', sum)
    print('sum_1', sum_1)
    if sum_1/znamen < sum/znamen:
        return True

    return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    indexes = []
    l = 20
    T = 20
    h = 1
    tau = 1
    a = 1

    l_h = int(l / h)
    T_tau = int(T / tau)

    u = [[[[15 for _ in range(T_tau + 1)] for j in range(l_h + 1)] for k in range(l_h + 1)] for i in range(l_h + 1)]

    np_u = np.array(u, dtype="float64")

    b = np.array([[[(i * i)/8 + (j * j)/8 + (k * k)/8 for k in range(l_h + 1)] for i in range(l_h + 1)] for j in range(l_h + 1)], dtype="float64")

    Xb = []
    Yb = []
    Zb = []
    for i in range(l_h + 1):
        for j in range(l_h + 1):
            Xb.append(i * h)
            Yb.append(j * h)
            Zb.append(b[i][j][0])

    Xu = []
    Yu = []
    Zu = []
    for i in range(l_h + 1):
        for j in range(l_h + 1):
            Xu.append(i * h)
            Yu.append(j * h)
            Zu.append(u[i][j][0][int(T / tau)])

    qk0 = np.array(u, dtype="float64")

    print("I start founding f")
    f = count_f(l, h, np_u, T, tau)
    #f = np.load('startf.npy')
    print(f)
    np.save('startf', f)

    Xf = []
    Yf = []
    Zf = []
    for i in range(l_h + 1):
        for j in range(l_h + 1):
            Xf.append(i * h)
            Yf.append(j * h)
            Zf.append(f[i][j][0][int(T / tau)])

    np_f = np.array(f, dtype="float64")

    print("I found start f")
    print('starting psi')
    psi = count_psi(l, h, b, np_f, T, tau, a)
    indices = np.where(psi == psi.max())
    print('max', psi[indices])  # prints [400]
    print("I found start psi")
    print(psi)
    #ak : float = 0.0000000000000000953085
    ak = 1 / 10000000000000000
    k = 0

    q_result = []

    fq1 = np.array([[[0 for _ in range(l_h + 1)] for _ in range(l_h + 1)] for i in range(l_h + 1)], dtype="float64")

    fq = np.array(f, dtype="float64")
    fq0 = np.zeros((int(l / h) + 1, int(l / h) + 1, int(l / h) + 1), dtype="float64")
    # fq0 = np.abs(fq[:, :, :, int(T / tau)] - b)

    for i in range(int(l / h) + 1):
        for j in range(int(l / h) + 1):
            for k in range(int(l / h) + 1):
                fq0[i][j][k] = fq[i][j][k][int(T / tau)] - b[i][j][k]
    o = 0
    print("I starting main circle")
    while o != 1:
        qk1 = count_q(l, h, qk0, psi, ak)

        fq = count_f(l, h, qk1, T, tau)
        maxNewf = 0

        for i in range(int(l / h) + 1):
            for j in range(int(l / h) + 1):
                for k in range(int(l / h) + 1):
                    fq1[i][j][k] = fq[i][j][k][int(T / tau)] - b[i][j][k]
                    if abs(fq1[i][j][k]) > maxNewf:
                        maxNewf = abs(fq1[i][j][k])

        print('start compare')
        print('max in new f', maxNewf)
        print('fq0 - b', fq0)
        print('fq1 - b -> 0', fq1)
        if not compare_control(fq0, fq1, T, tau, l, h):
            ak = ak / 2
            print("I not found new control, ", ak)
            continue

        psi = count_psi(l, h, b, fq, T, tau, a)

        indices = np.where(psi == psi.max())
        print('max', psi[indices])  # prints [400]

        print("I found new control, ", o)
        print("hurray")
        ak = 1 / 1000000000000000
        qk0 = np.array(qk1)
        fq0 = np.array(fq1)
        np.save('f' + str(o), fq0)
        q_result.append(qk1)
        o += 1

    for i in range(int(l / h) + 1):
        for j in range(int(l / h) + 1):
            for k in range(int(l / h) + 1):
                fq1[i][j][k] = fq[i][j][k][int(T / tau)]

    np.save('fq1', fq1)
    for i in range(0, len(q_result)):
        np.save('q' + str(i), q_result[i])

    print("f_result:", fq1)

    print("now=", datetime.now().time())

    Xq = []
    Yq = []
    Zq = []
    for i in range(l_h + 1):
        for j in range(l_h + 1):
            Xq.append(i * h)
            Yq.append(j * h)
            Zq.append(qk0[i][j][0][int(T / tau)])

    figu = plt.figure()
    axq = figu.add_subplot(211, projection='3d')
    axq.plot_trisurf(Xu, Yu, Zu, cmap=cm.jet, linewidth=0)
    axq.set_title("Начальное управление u")
    figu.tight_layout()

    figq = plt.figure()
    axq = figq.add_subplot(211, projection='3d')
    axq.plot_trisurf(Xq, Yq, Zq, cmap=cm.jet, linewidth=0)
    axq.set_title("Конечное управление q")
    figq.tight_layout()

    Xf1 = []
    Yf1 = []
    Zf1 = []
    for i in range(l_h + 1):
        for j in range(l_h + 1):
            Xf1.append(i * h)
            Yf1.append(j * h)
            Zf1.append(fq1[i][j][0])

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.plot_trisurf(Xb, Yb, Zb, cmap=cm.jet, linewidth=0)
    ax.set_title("Желаемое распределение тепла b")
    fig.tight_layout()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211, projection='3d')
    ax1.plot_trisurf(Xf1, Yf1, Zf1, cmap=cm.jet, linewidth=0)
    ax1.set_title("Конечное распределение тепла f")
    fig1.tight_layout()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(211, projection='3d')
    ax2.plot_trisurf(Xf, Yf, Zf, cmap=cm.jet, linewidth=0)
    ax2.set_title("Начальное распределение тепла f")
    fig2.tight_layout()

    plt.show()
