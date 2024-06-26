import numpy as np
import scipy.stats as sst
import os
import time
from MultiLevelPicardJump import MLP_model

path = os.path.join(os.getcwd(), "counterparty_jvasicek/")
dtype = np.float32

N = 12
Mnmax = 5
T = 1.0/2.0

x0 = 100.0
alpha = 0.01
mu0 = 100.0
sigma0 = 2.0
lam = 0.5
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
    def mu(t, x):
        return alpha*(mu0-x)
    
    def sigmadiag(t, x):
        return sigma0
    
    def eta(t, x, z):
        return z
    
    distr = sst.uniform(loc = 0.0, scale = 1.0)
    nuAdelta = lam*np.mean(np.linalg.norm(distr.rvs(size = [5000, d]), axis = -1) >= delta)
    def Zdelta(size):
        Z = distr.rvs(size = size)
        if np.prod(Z.shape) > 0.0:
            ind = np.linalg.norm(Z, axis = -1) < delta
            while np.sum(ind) > 0.0:
                Z[ind] = distr.rvs(size = [np.sum(ind), d])
                ind = np.linalg.norm(Z, axis = -1) < delta
                
        return Z.astype(np.float32)
    
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