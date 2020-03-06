import numpy as np
import scipy.integrate as integrate

# 1) Environment

lamb = np.array([[0.0, 0.2, 0.3],
                 [0.1, 0.0, 0.2],
                 [0.4, 0.0, 0.0]])

def sum_lambdas(lamb):
    sum_lambd = []
    for i in range(0, len(lamb)):
        sum_lambd.append(sum(lamb[i]))
    vector = sum_lambd
    sum_lambd = np.diag(sum_lambd)
    return vector, sum_lambd

vector, sum_lambd = sum_lambdas(lamb)

# 2) Regression

betta = np.array([[0.0, 2.0, 4.0],
                 [1.0, 3.0, 6.0],
                 [2.0, 5.0, 8.0]])

sigma = 1

ed = np.eye(4)

XF = np.transpose(np.array([[1 for i in range(30)],
                 [2, 3, 2, 2, 1, 1, 2, 3, 3, 1, 2, 3, 1, 1, 2, 3, 4, 2, 5, 3, 2, 3, 1, 3, 1, 2, 3, 4, 1, 1],
                 [1, 2, 4, 1, 3, 2, 3, 2, 3, 1, 1, 1, 2, 3, 3, 2, 1, 3, 3, 1, 2, 2, 2, 3, 4, 1, 2, 0.5, 2, 2]]))

tau_F = np.transpose(np.array([0.5, 0.8, 1, 0.6, 0.4, 1, 2, 0.4, 1, 0.4, 1, 0.4, 1, 1, 0.6, 0.7, 0.7, 0.5, 1, 0.6, 0.4, 0.9, 1, 2, 0.6, 0.5, 0.3, 1.3, 0.8, 1])) 

IF = np.transpose(np.array([1, 0, 2, 2, 1, 1, 0, 0, 1, 2, 1, 1, 0, 0, 2, 1, 1, 0, 2, 2, 0, 2, 2, 1, 0, 1, 2, 0, 0, 1]))

def randomize_X(XF):
    X = np.array([XF[:, 0], XF[:, 1] + np.random.uniform(-1, 1, len(XF[:, 1])),
                 XF[:, 2] + np.random.uniform(-0.5, 0.5, len(XF[:, 2]))])
    return X

X = np.transpose(randomize_X(XF))

def randomize_tau(tau_F):
    noise = np.random.uniform(0, 1, len(tau_F))
    tau = np.multiply(tau_F, 2*noise)
    return tau

tau = np.transpose(randomize_tau(tau_F))

stat_pr = [0.364, 0.242, 0.394]

def randomize_I(IF, stat_pr):
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

I = np.transpose(randomize_I(IF, stat_pr))

# 3 Modelling transition and responses Y

def Simtrans(ii):
    #J = 0
    if ii == 0:
        if np.random.uniform(0,1) < (lamb[0][1])/(lamb[0][1]+lamb[0][2]):
            J = 1
        else:
            J = 2
    elif ii == 1:
        if np.random.uniform(0,1) < (lamb[1][0])/(lamb[1][0]+lamb[1][2]):
            J = 0
        else:
            J = 2
    else:
        if np.random.uniform(0,1) < (lamb[2][0])/(lamb[2][0]+lamb[2][1]):
            J = 0
        else:
            J = 1
    return J

def SimY(tau, ii, x):
    t = 0
    I = ii
    Y = 0
    while t < tau:
        T = np.random.exponential(1/vector[I])
        bb = betta[:, I]
        tt = t + T
        Z = sigma*np.random.normal(0, 1)
        if tt < tau:
            delta = T
            t = tt
            I = Simtrans(I)
        else:
            delta = tau - t
            t = tt
        Y = Y + np.matmul(x, bb)*delta + np.sqrt(delta)*Z
    return Y
    
def SimYs(tau, II, X):
    Y = []
    for r in range(len(X)):
        xx = X[r]
        x = np.transpose(xx)
        t = tau[r]
        ii = II[r]
        Y.append(SimY(t, ii, x))
    return Y
        
def tau_in(tau):
    s = []
    for i in range(len(tau)):
        s.append(1/tau[i])
    return s


#5. Random environment
AA = np.transpose(lamb) - sum_lambd
#chi = np.linalg.eigvals(AA)
chi, M = np.linalg.eig(AA)
chi = np.around(chi, decimals=3)
M = np.around(M, decimals=3)
'''
print(np.around(np.matmul(AA, np.transpose(M)[0]), decimals=3))
print(np.around(np.matmul(AA, np.transpose(M)[1]), decimals=3))
print(np.around(np.matmul(AA, np.transpose(M)[2]), decimals=3))
print(np.around(chi[1]*np.transpose(M)[1], decimals=3))
print(np.around(chi[2]*np.transpose(M)[2], decimals=3))'''


def D_chi(chi, t):
    v = np.exp(t*chi)
    d = np.diag(v)
    return d

def Pr(M, t):    
    bf = np.linalg.inv(M)
    vf = np.matmul(M, D_chi(chi, t))
    tmp = [0 for i in range(len(vf))]
    R = []
    for i in range(len(bf)):
        for j in range(len(vf)):
            tmp += bf[:, i][j] * vf[:, j]
        R.append(tmp)
        tmp = [0 for i in range(len(vf))]
    return np.array(R)

#print(Pr(M, 100))

# 6 Expectation of a sojourn time

def expected_time(state, tt, M, chi, n=3):
    b = np.linalg.inv(M)[:, state]
    R = b[0]*M[:, 0]*tt
    for i in range(1, n):
        R += b[i]*(1/chi[i])*(np.exp(tt*chi[i]) -1)*M[:, i]
    return R
        
        
#np.kron(M, M)

def vec(A):
    A = np.transpose(A)
    return A.flatten()


def Ed(i, n):
    ed = [0 for j in range(n)]
    ed[i] = 1
    return ed


# expectation of the response

def expY(tt, ii, x):
    A = np.transpose(expected_time(ii, tt, M, chi, n=3))
    B = np.kron(A, x)
    R = np.matmul(B, vec(betta))
    return R

def expYs(t, I, X):
    R = []
    for i in range(0, len(tau_F)):
        x = np.transpose(X)[:, i]
        R.append(expY(t[i], I[i], x))
    return R


# preparing
# regressor matrix Xreg

def Xreg(tau, I, X):
    R = []
    for i in range(0, len(tau)):
        ii = I[i]
        tt = tau[i]
        A = np.transpose(expected_time(ii, tt, M, chi, n=3))
        C = np.transpose(X)[:, i]
        B = np.transpose(C)
        R.append(np.transpose(np.kron(A, B)))
    return np.array(R)

# 10 estimation

def betta_est2(tau, I, X, Y, W):
    MX = Xreg(tau, I, X)
    MXX = np.matmul(np.matmul(np.transpose(MX), W), MX)
    VY = np.matmul(np.matmul(np.transpose(MX), W), Y)
    betta = np.matmul(np.linalg.inv(MXX), VY)
    return betta
 
YOur = [5.655, 9.961, 47.034, 14.539, 8.251, 14.595, 32.249, 3.801, 26.521, 7.784, 14.161, 6.741, 3.85, 17.298, 22.717, 14.588,
        24.809, 4.493, 56.951, 18.758, 2.865, 34.465, 21.864, 53.517, 7.056, 4.778, 11.531, 37.583, 10.78, 15.237] 
  
#print(betta_est2(tau_F, IF, XF, YOur, np.eye(30)))
#print(betta_est2(tau_F, IF, XF, YOur, np.diag(tau_in(tau_F))))    
    
#Pr(M, t)
#expected_time(state, tt, M, chi, n=3)
def mu2(i, j, L, t):
    def to_integrate(z):
        return (Pr(M, z)[i, j]*expected_time(j, t-z, M, chi, n=3)[L]) + (Pr(M, z)[i, L]*expected_time(L, t-z, M, chi, n=3)[j])
    result = integrate.quad(to_integrate, 0, t)
    return result

#print(mu2(1,1,1,1))
    
def cov_mu2(i, t, n=3):
    tt = expected_time(i, t, M, chi, n=3)
    C = [[0 for k in range(n)] for z in range(n)]
    for j in range(n):
        for L in range(n):
            if j == L:
                C[j][L] = mu2(i,j,j,t)[0] - tt[j]*tt[j]
            else:
                C[j][L] = mu2(i,j,L,t)[0] - tt[j]*tt[L]
    return np.array(C)
        
#print(cov_mu2(1, 10))
def varY(vecb_est, sigma_est, i, x, t):
    varY = (sigma_est**2)*t + np.matmul(np.matmul(np.transpose(vecb_est), np.kron(cov_mu2(i, t),np.matmul(np.transpose(np.matrix(x)), np.matrix(x)))), vecb_est) 
    return float(varY)

#print(np.matmul(np.transpose(np.matrix(XF[2])), np.matrix(XF[2])))
#print(varY(vec(betta), 1, 2, XF[2], 1))

def WLast(vecb_est, sigma_est, i, x, tau):
    V = []
    for n in range(len(tau)):
        V.append(1/(varY(vecb_est, sigma_est, i[n], XF[n], tau[n])))
    return np.diag(V)        
        
#print(WLast(vec(betta), 1, IF, XF, tau_F))

#13 Improvement of the estimators

#print(betta_est2(tau_F, IF, XF, YOur, WLast(vec(betta), 1, IF, XF, tau_F)))


#14 Estimation of sigma

def sigma_est2(I, X, tau, VecBest, Y):
    XT = X
    RSS = 0.0
    RRR = 0.0
    tau_sum = 0.0
    for i in range(len(I)):
        tau_sum += tau[i]
        t_n = np.transpose(expected_time(I[i], tau[i], M, chi, n=3))
        xn = XT[i]
        S = np.kron(np.transpose(np.matrix(t_n)), np.transpose(np.matrix(xn)))
        #print('S', float(S))
        RSS += (Y[i] - float(S[0]))**2
        cov = cov_mu2(I[i], tau[i], n=3)
        XTX = np.matmul(np.matrix(xn), np.transpose(np.matrix(xn)))
        RRR += np.kron(np.transpose(np.matrix(cov)), np.transpose(np.matrix(XTX)))
    print(1./tau_sum)
    print('rss', RSS)
    print(VecBest)
    print(VecBest*RRR)
    print(np.matmul(np.transpose(VecBest)*RRR, VecBest))
    R = (1./tau_sum)*(RSS - np.matmul(np.transpose(VecBest)*RRR, VecBest))
    return R

print(sigma_est2(IF, XF, tau_F, vec(betta), YOur))


    



