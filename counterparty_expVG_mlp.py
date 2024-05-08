import numpy as np
import scipy.special as ssp
import os
import time
from MultiLevelPicardJump import MLP_model

path = os.path.join(os.getcwd(), "counterparty_expVG/")
dtype = np.float32

N = 12
Mnmax = 5
T = 1.0/2.0

x0 = 100.0
mu0 = -0.0001
sigma0 = 0.01
alpha = 0.1
kappa = 0.0001
delta = 0.1
M0 = 200

beta = 0.03
def f(t, x, y):
    return -beta*np.fmin(y, 0)

K1 = 80.0
K2 = 100.0
L = 5.0
def g(x):
    return np.fmax(np.min(x, axis = -1, keepdims = True) - K1, 0) - np.fmax(np.min(x, axis = -1, keepdims = True) - K2, 0) - L

dd = [10, 50, 100, 500, 1000, 5000, 10000]
runs = 10

print("======================================================================")
for i in range(len(dd)):
    d = dd[i]
    
    c = (0.2168+0.932*d/2.0)/(0.392+d/2.0)
    gamma = 2.0*np.power(d, c)/(1.0+np.power(d, c))
    lambd = gamma*np.sqrt(np.pi)*np.exp(ssp.gammaln(d/2.0+0.5)-ssp.gammaln(d/2.0))/ssp.gamma(1.0/gamma)
    
    # Approximate K_{d/2}(x) by the function in https://arxiv.org/abs/2303.13400
    def CDF(x):
        exp1 = ssp.exp1(np.power(np.sqrt(2*alpha/kappa)*x/lambd, gamma))
        return 2*alpha*exp1/gamma
    
    nuAdelta = CDF(delta)
    
    # Approximate the inverse of E1(x) by the following function
    # see also https://mathematica.stackexchange.com/questions/251068/asymptotic-inversion-of-expintegralei-function
    eulermasc = 0.5772156649
    def invE1(x):
        y1 = -np.log(x)-np.log(-np.log(x))-(np.log(-np.log(x))-1.0)/np.log(x)
        y2 = np.exp(-(x+eulermasc))+np.exp(-2.0*(x+eulermasc))+1.25*np.exp(-3.0*(x+eulermasc))
        y1 = np.expand_dims((x < 0.2043338275)*y1, -1)
        y2 = np.expand_dims((x >= 0.2043338275)*y2, -1)
        y = np.concatenate([y1, y2], axis = -1)
        return np.nansum(y, axis = -1)
    
    def invCDF(x):
        y = np.power(invE1(nuAdelta*gamma*x/(2.0*alpha)), 1/gamma)
        return np.sqrt(kappa/(2.0*alpha))*lambd*y
    
    # We split up Z ~ \nu^d_\delta(dz) into Z ~ R * V, where:
    # the random radius R ~ invCDF(U) is obtained using the inverse transform sampling with U ~ Unif(0,1) (see https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    # the random direction V ~ Unif(S^{d-1}) is uniformly distributed on the sphere S^{d-1} (see https://dl.acm.org/doi/pdf/10.1145/377939.377946)
    def Zdelta(size):
        Y = np.random.normal(size = size)
        V = Y/np.linalg.norm(Y, axis = -1, keepdims = True)
        U = np.random.uniform(low = 0.0, high = 1.0, size = np.append(size[:-1], 1))
        R = invCDF(U)
        Z = R*V
        return Z.astype(np.float32)
    
    # We approximate I_nu by sampling random variables Z from the truncated Levy measure
    Z = Zdelta(size = [5000, d])
    Inu = nuAdelta*np.mean(np.exp(Z)-1.0)
    
    def mu(t, x):
        return (mu0+sigma0**2/2+Inu)*x
    
    def sigmadiag(t, x):
        return sigma0*x
    
    def eta(t, x, z):
        return np.expand_dims(x, 0)*(np.exp(z)-1.0)
    
    sol = np.zeros([Mnmax, runs])
    tms = np.zeros([Mnmax, runs])
    fev = np.zeros([Mnmax, runs])
    for M in range(1, Mnmax+1):
        for r in range(runs):
            b = time.time()
            mlp = MLP_model(d, M, N, T, mu, sigmadiag, eta, Zdelta, M0, nuAdelta, f, g)
            x1 = x0*np.ones(d, dtype = dtype)
            sol[M-1, r] = mlp.compute(0.0, x1, M)
            e = time.time()
            tms[M-1, r] = e-b
            fev[M-1, r] = mlp.counts
            print("MLP performed for d = " + str(d) + ", m = " + str(M) + "/" + str(Mnmax) + ", run " + str(r+1) + "/" + str(runs) + ", in " + str(np.round(tms[M-1, r], 1)) + "s with solution " + str(sol[M-1, r]))
       
            np.savetxt(path + "mlp_sol_" + str(dd[i]) + ".csv", sol)
            np.savetxt(path + "mlp_tms_" + str(dd[i]) + ".csv", tms)
            np.savetxt(path + "mlp_fev_" + str(dd[i]) + ".csv", fev)
    
print("======================================================================")
print("MLP solutions saved")