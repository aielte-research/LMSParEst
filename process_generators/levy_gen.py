import numpy as np

#SOURCE: https://github.com/k945743/alpha_stab_levy__sym_disc

#### comments on implementation: ####

#constraints: 
# time_zero == 0
# seq_len = len(G) = len(W)
# alpha \in (0,2]

# core method:
""" def alpha_stab_levy(times, alpha, G,W ):
    inc=[0]
    N=len(times)
    for i in range(N-1):
        inc.append((times[i+1]-times[i])**(1/alpha)*(np.sin(alpha*G[i]))/((np.cos(G[i]))**(1/alpha))*((np.cos((1-alpha)*G[i]))/(W[i]))**((1-alpha)/(alpha)))
    return np.cumsum(inc) """

#EXAMPLE:  direct usage of the method "alpha_stab_levy" above
""" time_zero = 0
horizon = 1
seq_len = 100
times=[i*horizon/seq_len for i in range(time_zero,seq_len)]
low=-np.pi/2
high=np.pi/2
G=np.random.uniform(low,high,seq_len)
W=np.random.exponential(1,seq_len)
alpha = 1.1
path=alpha_stab_levy(times,alpha,G,W)
#plt.plot(path) """
####                              ####


class AlphaStabSymLevy():
    def __init__(self,alpha,n=1000,T=1,times=None):
        if times is None:
            self.seq_len = n
            self.time_horizon = T
            self.times = [i*self.time_horizon/self.seq_len for i in range(self.seq_len)]
        else:
            self.times = times
            self.seq_len = len(self.times)-1
           
        self.alpha = alpha
    
    def alpha_stab_levy(self, times, alpha, G,W ):
        inc=[0]
        N=len(times)
        for i in range(N-1):
            inc.append((times[i+1]-times[i])**(1/alpha)*(np.sin(alpha*G[i]))/((np.cos(G[i]))**(1/alpha))*((np.cos((1-alpha)*G[i]))/(W[i]))**((1-alpha)/(alpha)))
        return np.cumsum(inc)
    
    def __call__(self):
        low=-np.pi/2
        high=np.pi/2
        G=np.random.uniform(low,high,self.seq_len)
        W=np.random.exponential(1,self.seq_len)
        path = self.alpha_stab_levy(self.times,self.alpha,G,W)
        return path


#EXAMPLES:

#import matplotlib.pyplot as plt

#EXAMPLE 1:

#levy = AlphaStabSymLevy([1000,1],1.3)
#plt.plot(levy())

#EXAMPLE 2:

#horizon = 1
#seq_len = 1000
#times=[i*horizon/seq_len for i in range(seq_len)]
#levy_t = AlphaStabSymLevy(times,1.3,switch="times")
#plt.plot(levy_t())

def gen(n, alpha=1, **params):
    generator=AlphaStabSymLevy(alpha=alpha,n=n)
    return generator()