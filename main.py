import mmlr
import matplotlib.pyplot as plt
import numpy as np


def experiment():
    est = mmlr.Estimation()
    XF = np.transpose(np.array([[1 for i in range(30)],
                                [3, 2, 2, 1, 1, 1, 2, 1, 3, 1, 2, 3, 1, 1, 2, 3, 4, 2, 5, 3, 2, 1, 1, 3, 1, 2, 3, 4, 1,
                                 1],
                                [1.1, 2.2, 4.3, 1.2, 3, 2, 3, 2, 3.2, 1, 1, 1.5, 2, 3.3, 4, 2, 1, 3, 3.2, 1, 3.1, 2, 4,
                                 3, 4, 1, 2, 0.5, 2, 2]]))
    print(XF)
    space = np.linspace(0.05, 0.55, 10)  # from, to, how many elements
    brezs_mean = []
    pes_mean = []
    r2s_means = []
    rmses_means = []
    Fs_means = []
    for i in range(10):
        print('-----------------------Step ', i)
        mult = space[i]
        brezs = [0 for i in range(9)]
        pes = []
        r2s = []
        rmses = []
        Fs = []
        for i in range(10):
            brez, pe, r_squared, rmse, F = est.iterations(600, est.reg.sigma, est.reg.betta, XF, wlast=False,
                                                          p1=-10 * mult, p2=10 * mult, p3=-1 * mult, p4=1 * mult)
            brezs += brez
            pes.append(pe)
            r2s.append(r_squared)
            rmses.append(rmse)
            Fs.append(F)

        brezs_mean.append(brezs / 10)
        pes_mean.append(np.mean(pes))

        r2s_means.append(np.mean(r2s))
        rmses_means.append(np.mean(rmses))
        Fs_means.append(np.mean(Fs))

    print(brezs_mean)
    print(pes_mean)
    plt.scatter(space, rmses_means, marker='o', s=60, color='green')
    plt.plot(space, rmses_means, color='green')
    plt.xlabel('Noise factor')
    plt.ylabel('RMSE')
    plt.grid()
    plt.show()

    plt.scatter(space, r2s, marker='o', s=60, color='green')
    plt.plot(space, r2s, color='green')
    plt.xlabel('Noise factor')
    plt.ylabel('Coefficient of determination')
    plt.grid()
    plt.show()

    plt.scatter(space, Fs, marker='o', s=60, color='green')
    plt.plot(space, Fs, color='green')
    plt.xlabel('Noise factor')
    plt.ylabel('F-static')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    experiment()