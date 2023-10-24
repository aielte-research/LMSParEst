import random
import numpy as np

def rand_ar_params(length):
    counter = 0
    poly = [1]
    while counter < length:
        if (counter == length - 1) or (random.randint(0, 1) == 0):
            factor = [1, random.uniform(-1, 1)]
            counter += 1
        else:
            factor = [1, 0, random.uniform(0, 1)]
            counter += 2

        poly = np.convolve(poly, factor)
        poly = list(poly)


    #poly = [-coef / poly[-1] for coef in poly]
    poly = poly[1:]
    out = [-coef for coef in poly]

    return out

def rand_rand_ar_params(max_length):
    length = random.randint(1, max_length)
    
    out = rand_ar_params(length)

    return out


def rand_unif_list(length, a, b):
    out = [random.uniform(a, b) for _ in range(length)]

    return out

def rand_rand_unif_list(max_length, a, b):

    length = random.randint(1, max_length)
    out = rand_unif_list(length, a, b)

    return out

    
#x = rand_rand_ar_params(1)
#print(x)


