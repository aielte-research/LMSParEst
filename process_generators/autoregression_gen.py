import numpy as np

def gen(n, alpha=1, distr={"name":"uniform", "low":0, "high":1}, **params):
    if distr["name"]=="uniform":
        S = np.random.uniform(low=distr["low"], high=distr["high"], size=n)
    elif distr["name"]=="normal":
        S = np.random.normal(loc=distr["loc"], scale=distr["scale"], size=n)
    else:
        raise ValueError("Distribution unknown: " + distr["name"])
        
    ret = [S[0]]
    for x in S[1:]:
        ret.append(alpha*ret[-1]+x)
    return ret
