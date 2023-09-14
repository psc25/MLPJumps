import numpy as np

class MLP_model():
    def __init__(self, d, M, N, T, mu, sigmadiag, eta, Zdistr, M0, nuAdelta, f, g, dtype = np.float32):
        self.d = d
        self.M = M
        self.N = N
        self.T = T
        self.mu = mu
        self.sigmadiag = sigmadiag
        self.eta = eta
        self.Zdistr = Zdistr
        self.M0 = M0
        self.nuAdelta = nuAdelta
        self.f = f
        self.g = g
        self.counts = np.zeros(shape = 1, dtype=np.int64)
        self.dtype = dtype
    
    def compute(self, t, x, n):
        if n == 0:
            return 0.0
        else:
            a = 0.0
            dt = (self.T-t)/self.N
            Mn = np.power(self.M, n)
            for i in range(Mn):
                X = x
                W = np.random.normal(size = [self.N, self.d], scale = np.sqrt(dt)).astype(self.dtype)
                P = np.random.poisson(size = self.N, lam = self.nuAdelta*dt)
                Z = self.Zdistr.rvs(size = [self.N, np.max(P), self.d]).astype(self.dtype)
                V = self.Zdistr.rvs(size = [self.N, self.M0, self.d]).astype(self.dtype)
                self.counts = self.counts + self.N*self.d + self.N + self.N*np.max(P)*self.d + self.N*self.M0*self.d
                for k in range(self.N):
                    drift = self.mu(t, X)*dt
                    diffu = self.sigmadiag(t, X)*W[k]
                    jump1 = np.sum(self.eta(t, X, Z[k, 0:P[k]]), axis = 0)
                    jump2 = self.nuAdelta*dt*np.mean(self.eta(t, X, V[k]), axis = 0)
                    X = X + drift + diffu + jump1 - jump2
                    self.counts = self.counts + 2 + P[k] + self.M0
                    
                a = a + self.g(X)
                self.counts = self.counts + 1
                
            u = a/Mn
            
            for l in range(n):
                Mnl = np.power(self.M, n-l)
                b = 0.0
                for i in range(Mnl):
                    Y = x
                    R = np.random.uniform(t, self.T)
                    S = np.int32(np.floor((R-t)/dt) + 1)
                    W = np.random.normal(size = [S, self.d], scale = np.sqrt(dt)).astype(self.dtype)
                    P = np.random.poisson(size = S, lam = self.nuAdelta*dt)
                    dt2 = R - (t + (S-1)*dt)
                    P = np.append(P, np.random.poisson(lam = self.nuAdelta*dt2))
                    Z = self.Zdistr.rvs(size = [S, np.max(P), self.d]).astype(self.dtype)
                    V = self.Zdistr.rvs(size = [S, self.M0, self.d]).astype(self.dtype)     
                    self.counts = self.counts + 1 + S*self.d + (S+1) + S*np.max(P)*self.d + S*self.M0*self.d
                    for k in range(S):
                        drift = self.mu(t, Y)*dt
                        diffu = self.sigmadiag(t, Y)*W[k]
                        jump1 = np.sum(self.eta(t, Y, Z[k, 0:P[k]]), axis = 0)
                        jump2 = self.nuAdelta*dt*np.mean(V[k], axis = 0)
                        Y = Y + drift + diffu + jump1 - jump2
                        self.counts = self.counts + 2 + P[k] + self.M0
                    
                    b = b + self.f(R, Y, self.compute(R, Y, l))
                    self.counts = self.counts + 1
                    if l > 0:
                        b = b - self.f(R, Y, self.compute(R, Y, l-1))
                        self.counts = self.counts + 1
                        
                u = u + (self.T-t)*b/Mnl
                
            return u