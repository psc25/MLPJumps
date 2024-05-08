import numpy as np
import os
import time
from MultiLevelPicard import MLP_model

path = os.path.join(os.getcwd(), "default_bs\\")
dtype = np.float32

N = 12
Mnmax = 5
T = 1.0/3.0

mu0 = -0.01
sigma0 = 0.15

beta, R = 2.0/3.0, 0.02
vh, vl = 25.0, 50.0
gammah, gammal = 0.2, 0.02
def f(t, x, y):
    return -(1.0-beta)*np.fmin(np.fmax((y-vh)*(gammah-gammal)/(vh-vl)+gammah, gammal), gammah)*y - R*y

def g(x):
    return np.min(x, axis = -1, keepdims = True)

dd = [10, 50, 100, 500, 1000, 5000, 10000]
runs = 10

print("======================================================================")
for i in range(len(dd)):
    d = dd[i]
    def mu(t, x):
        return (mu0+sigma0**2/2)*x
    
    def sigmadiag(t, x):
        return sigma0*x
    
    sol = np.zeros([Mnmax, runs])
    tms = np.zeros([Mnmax, runs])
    fev = np.zeros([Mnmax, runs])
    for M in range(1, Mnmax+1):
        for r in range(runs):
            b = time.time()
            mlp = MLP_model(d, M, N, T, mu, sigmadiag, f, g)
            x0 = 30.0*np.ones(d, dtype = dtype)
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