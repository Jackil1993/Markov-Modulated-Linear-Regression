import numpy as np
import scipy.integrate as integrate


class Environment:
    def __init__(self):
        self.lamb = np.array([[0.0, 0.2, 0.3],
                         [0.1, 0.0, 0.2],
                         [0.4, 0.0, 0.0]])
        self.vector, self.sum_lambd = self.sum_lambdas(self.lamb)

    def sum_lambdas(self, lamb):
        sum_lambd = []
        for i in range(0, len(lamb)):
            sum_lambd.append(sum(lamb[i]))
        vector = sum_lambd
        sum_lambd = np.diag(sum_lambd)
        return vector, sum_lambd


class Regression:
    def __init__(self):
        self.betta = np.array([[2.1, 3.7, 4.2],
                               [1.7, 3.2, 6.3],
                               [5.2, 8.4, 3.0]])
        self.sigma = 2
        self.ed = np.eye(4)
        self.tau_F = np.transpose(np.array([0.4, 0.9, 1.1, 0.5, 0.4, 0.9, 2, 0.4, 1, 0.4, 1.2, 0.4, 1, 1.1, 0.6,
                                            0.7, 0.7, 0.6, 1, 0.7, 0.4, 0.9, 1, 2.1, 0.6, 0.5, 0.3, 1.3, 0.8, 1.2]))
        self.IF = np.transpose(np.array([1, 0, 2, 2, 1, 1, 0, 0, 1, 2, 1, 1, 0, 0, 2, 1, 1, 0, 2, 2, 0, 2, 2, 1, 0, 1, 2, 0, 0, 1]))
        self.stat_pr = [0.364, 0.242, 0.394] #stat_pr
    def randomize_X(self, XF, p1=-1, p2=1, p3=-0.5, p4=0.5):
        X = np.array([XF[:, 0], XF[:, 1] + np.random.uniform(p1, p2, len(XF[:, 1])),
                     XF[:, 2] + np.random.uniform(p3, p4, len(XF[:, 2]))])
        return X

    def randomize_tau(self, tau_F):
        noise = np.random.uniform(0, 1, len(tau_F))
        tau = np.multiply(tau_F, 2*noise)
        return tau

    def randomize_I(self, IF, stat_pr):
        I = []
        for i in range(len(IF)):
            rnd = np.random.uniform(0, 1)
            if rnd < stat_pr[0]:
                I.append(0)
            elif stat_pr[0] < rnd < stat_pr[0] + stat_pr[1]:
                I.append(1)
            else:
                I.append(2)
        return I

# modelling transition and responses Y

class Modelling:
    def __init__(self):
        self.env = Environment()
        self.reg = Regression()
        self.lamb = self.env.lamb
        self.vector = self.env.vector
        self.sigma = self.reg.sigma
        self.tau_F = self.reg.tau_F
        # Random environment
        self.AA = np.transpose(self.lamb) - self.env.sum_lambd
        self.chi, self.M = np.linalg.eig(self.AA)
        self.chi = np.around(self.chi, decimals=3)
        self.M = np.around(self.M, decimals=3)

    def Simtrans(self,ii):
        if ii == 0:
            if np.random.uniform(0,1) < (self.lamb[0][1])/(self.lamb[0][1]+self.lamb[0][2]):
                J = 1
            else:
                J = 2
        elif ii == 1:
            if np.random.uniform(0,1) < (self.lamb[1][0])/(self.lamb[1][0]+self.lamb[1][2]):
                J = 0
            else:
                J = 2
        else:
            if np.random.uniform(0,1) < (self.lamb[2][0])/(self.lamb[2][0]+self.lamb[2][1]):
                J = 0
            else:
                J = 1
        return J

    def SimY(self, tau, ii, x, betta):
        t = 0
        I = ii
        Y = 0
        while t < tau:
            T = np.random.exponential(1/self.vector[I])
            bb = betta[:, I]
            tt = t + T
            Z = self.sigma*np.random.normal(0, 1)
            if tt < tau:
                delta = T
                t = tt
                I = self.Simtrans(I)
            else:
                delta = tau - t
                t = tt
            Y = Y + np.matmul(x, bb)*delta + np.sqrt(delta)*Z
        return Y
    
    def SimYs(self, tau, II, X, Betta):
        Y = []
        for r in range(len(X)):
            xx = X[r]
            x = np.transpose(xx)
            t = tau[r]
            ii = II[r]
            Y.append(self.SimY(t, ii, x, Betta.reshape(-1, 3)))
        return Y
        
    def tau_in(self, tau):
        s = []
        for i in range(len(tau)):
            s.append(1/tau[i])
        return s

    def D_chi(self, chi, t):
        v = np.exp(t*chi)
        d = np.diag(v)
        return d

    def Pr(self, M, t):
        bf = np.linalg.inv(M)
        vf = np.matmul(M, self.D_chi(self.chi, t))
        tmp = [0 for i in range(len(vf))]
        R = []
        for i in range(len(bf)):
            for j in range(len(vf)):
                tmp += bf[:, i][j] * vf[:, j]
            R.append(tmp)
            tmp = [0 for i in range(len(vf))]
        return np.array(R)

    # expectation of a sojourn time

    def expected_time(self, state, tt, M, chi, n=3):
        b = np.linalg.inv(M)[:, state]
        R = b[0]*M[:, 0]*tt
        for i in range(1, n):
            R += b[i]*(1/chi[i])*(np.exp(tt*chi[i]) -1)*M[:, i]
        return R

    def vec(self, A):
        A = np.transpose(A)
        return A.flatten()

    def Ed(self, i, n):
        ed = [0 for j in range(n)]
        ed[i] = 1
        return ed

    # expectation of the response

    def expY(self, tt, ii, x, Betta):
        A = np.transpose(self.expected_time(ii, tt, self.M, self.chi, n=3))
        B = np.kron(A, x)
        R = np.matmul(B, self.vec(Betta))
        return R

    def expYs(self, t, I, X, Betta):
        R = []
        for i in range(0, len(self.tau_F)):
            x = np.transpose(X)[:, i]
            R.append(self.expY(t[i], I[i], x, Betta))
        return R

    # preparing regressor matrix Xreg

    def Xreg(self, tau, I, X):
        R = []
        for i in range(0, len(tau)):
            ii = I[i]
            tt = tau[i]
            A = np.transpose(self.expected_time(ii, tt, self.M, self.chi, n=3))
            C = np.transpose(X)[:, i]
            B = np.transpose(C)
            R.append(np.transpose(np.kron(A, B)))
        return np.array(R)

class Estimation():
    def __init__(self):
        self.model = Modelling()
        self.env = Environment()
        self.reg = Regression()

        self.YOur = [5.655, 9.961, 47.034, 14.539, 8.251, 14.595, 32.249,
                     3.801, 26.521, 7.784, 14.161, 6.741, 3.85, 17.298, 22.717, 14.588,
                     24.809, 4.493, 56.951, 18.758, 2.865, 34.465, 21.864, 53.517,
                     7.056, 4.778, 11.531, 37.583, 10.78, 15.237]

    def betta_est2(self, tau, I, X, Y, W):
        MX = self.model.Xreg(tau, I, X)
        MXX = np.matmul(np.matmul(np.transpose(MX), W), MX)
        VY = np.matmul(np.matmul(np.transpose(MX), W), Y)
        betta = np.matmul(np.linalg.inv(MXX), VY)
        return betta

    def mu2(self, i, j, L, t):
        def to_integrate(z):
            return (self.model.Pr(self.model.M, z)[i, j]*self.model.expected_time(j, t-z, self.model.M, self.model.chi, n=3)[L]) + \
                   (self.model.Pr(self.model.M, z)[i, L]*self.model.expected_time(L, t-z, self.model.M, self.model.chi, n=3)[j])
        result = integrate.quad(to_integrate, 0, t)
        return result

    def cov_mu2(self, i, t, n=3):
        tt = self.model.expected_time(i, t, self.model.M, self.model.chi, n=3)
        C = [[0 for k in range(n)] for z in range(n)]
        for j in range(n):
            for L in range(n):
                if j == L:
                    C[j][L] = self.mu2(i,j,j,t)[0] - tt[j]*tt[j]
                else:
                    C[j][L] = self.mu2(i,j,L,t)[0] - tt[j]*tt[L]
        return np.array(C)

    def varY(self, vecb_est, sigma_est, i, x, t):
        varY = (sigma_est**2)*t + np.matmul(np.matmul(np.transpose(vecb_est), np.kron(self.cov_mu2(i, t), np.matmul(np.transpose(np.matrix(x)), np.matrix(x)))), vecb_est)
        return float(varY)

    def WLast(self, vecb_est, sigma_est, i, x, tau, XF):
        V = []
        for n in range(len(tau)):
            V.append(1/(self.varY(vecb_est, sigma_est, i[n], XF[n], tau[n])))
        return np.diag(V)

    # improvement of the estimators

    def improve(self, XF):
        return self.betta_est2(self.reg.tau_F, self.reg.IF, XF, self.YOur,
                               self.WLast(self.model.vec(self.reg.betta), 1, self.reg.IF, XF, self.reg.tau_F))

    # estimation of sigma

    def sigma_est2(self, I, X, tau, VecBest, Y):
        XT = X
        RSS = 0.0
        RRR = 0.0
        tau_sum = 0.0
        for i in range(len(I)):
            tau_sum += tau[i]
            t_n = self.model.expected_time(I[i], tau[i], self.model.M, self.model.chi, n=3)
            xn = XT[i]
            S = np.matmul(np.kron(np.matrix(t_n), np.matrix(xn)), np.transpose(np.matrix(VecBest)))
            RSS += (Y[i] - float(S[0]))**2
            cov = self.cov_mu2(I[i], tau[i], n=3)
            XTX = np.matmul(np.transpose(np.matrix(xn)), np.matrix(xn))
            RRR += np.kron(np.transpose(np.matrix(cov)), np.transpose(np.matrix(XTX)))
        R = (1./tau_sum)*(RSS - np.matmul(np.matmul(np.matrix(VecBest), np.matrix(RRR)), np.transpose(np.matrix(VecBest))))
        return float(R)

    # iterations

    def iterations(self, PP, sigma, best, XF, p1=-1, p2=1, p3=-0.5, p4=0.5, wlast=True):
        bvec = self.model.vec(best)
        rk = len(best)
        brez = [0 for i in range(len(bvec))]
        sigma_rez = 0
        l = len(XF)
        for p in range(0, PP):
            Xp = np.transpose(self.reg.randomize_X(XF, p1, p2, p3, p4))
            tt = np.transpose(self.reg.randomize_tau(self.reg.tau_F))
            II = np.transpose(self.reg.randomize_I(self.reg.IF, self.reg.stat_pr))
            Yp = self.model.SimYs(tt, II, Xp, self.reg.betta)
            if wlast == True:
                W = self.WLast(bvec, sigma, II, Xp, tt)
            else:
                W = np.eye(l)
            Bp = self.betta_est2(tt, II, Xp, Yp, W)
            brez += Bp
        brez = (1./PP)*brez
        pred = self.model.SimYs(self.reg.tau_F, self.reg.IF, XF, brez)
        Y = self.model.SimYs(self.reg.tau_F, self.reg.IF, XF, self.reg.betta)

        def calculate_error(y, pr):
            rss = 0.0
            mean = np.mean(y)
            tss = 0.0
            pe = 0.0
            ess = 0.0
            for i in range(len(y)):
                rss += (y[i] - pr[i])**2
                tss += (y[i] - mean)**2
                pe += (abs(y[i] - pr[i])/y[i])*100
                ess += (pr[i] - mean)**2
            r_squared = 1 - (rss/tss)
            mape = pe/len(y)
            rmse = np.sqrt(rss/len(y))
            F = (ess/9)/(rss/21)
            return r_squared, rmse, F #, mape
        r_squared, rmse, F = calculate_error(Y, pred)
        pe = 0.0
        be = self.reg.betta.flatten()
        for i in range(len(brez)):
            pe += (abs(be[i] - brez[i])/be[i])*100

        return brez, pe/len(brez), r_squared, rmse, F
