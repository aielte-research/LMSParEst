from databases.fbm_gen import gen as fbm

def gen(n = 200, hurst1 = 0.5, lambd1 = 1, hurst2 = 0.5, lambd2 = 1):

    fbm1 = fbm(n = n, hurst = hurst1, lambd = lambd1)
    fbm2 = fbm(n = n, hurst = hurst2, lambd = lambd2)

    return fbm1 + fbm2

