import numpy as np
import scipy.stats as sst
import os
import time
from MultiLevelPicardJump import MLP_model

path = os.path.join(os.getcwd(), "default_merton/")
dtype = np.float32

N = 12
Mnmax = 4
T = 1.0/3.0

mu0 = 0.02
sigma0 = 0.2
lam = 0.5
muZ = -0.03
sigmaZ = 0.2
delta = 0.1
M0 = 200

beta, R = 2.0/3.0, 0.02
vh, vl = 50.0, 70.0
gammah, gammal = 0.2, 0.02
def f(t, x, y):
    return -(1.0-beta)*np.fmin(np.fmax((y-vh)*(gammah-gammal)/(vh-vl)+gammah, gammal), gammah)*y - R*y

def g(x):
    return np.min(x, axis = -1, keepdims = True)

dd = [10000]
runs = 10

print("======================================================================")
for i in range(len(dd)):
    d = dd[i]
    Zdistr = sst.norm(loc = muZ, scale = sigmaZ)
    nuAdelta = lam*np.mean(np.linalg.norm(Zdistr.rvs(size = [5000, d]), axis = -1) >= delta)
    
    def mu(t, x):
        return (mu0+sigma0**2/2+lam*(np.exp(muZ+sigmaZ**2/2)-1-muZ))*x
    
    def sigmadiag(t, x):
        return sigma0*x
    
    def eta(t, x, z):
        return np.expand_dims(x, 0)*(np.exp(z)-1)
    
    sol = np.zeros([Mnmax, runs])
    tms = np.zeros([Mnmax, runs])
    fev = np.zeros([Mnmax, runs])
    for M in range(1, Mnmax+1):
        for r in range(runs):
            b = time.time()
            mlp = MLP_model(d, M, N, T, mu, sigmadiag, eta, Zdistr, M0, nuAdelta, f, g)
            x0 = 50.0*np.ones(d, dtype = dtype)
            sol[M-1, r] = mlp.compute(0.0, x0, M)
            e = time.time()
            tms[M-1, r] = e-b
            fev[M-1, r] = mlp.counts
            print("MLP performed for d = " + str(d) + ", m = " + str(M) + "/" + str(Mnmax) + ", run " + str(r+1) + "/" + str(runs) + ", in " + str(np.round(tms[M-1, r], 1)) + "s with solution " + str(sol[M-1, r]))
       
        np.savetxt(path + "mlp_sol_" + str(dd[i]) + ".csv", sol)
        np.savetxt(path + "mlp_tms_" + str(dd[i]) + ".csv", tms)
        np.savetxt(path + "mlp_fev_" + str(dd[i]) + ".csv", fev)
    
print("======================================================================")
print("MLP solutions saved")