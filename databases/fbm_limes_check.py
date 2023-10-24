import time
from fbm_generators import fbm
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import ctypes
import pandas as pd


def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


def diff_zero(hurst, seq_range, seq_step, number_of_random_outcomes, method="fft"):
    """
        Numerical test assingned to the issue Limes check

        Inputs:
        hurst: the Hurst exponent, [0,1]
        seq_range: the range of the length the sequences will be generated according to
        seq_step: the step of the range  seq_range
        number_of_random_outcomes: the number of possible outcomes from the sample space
        method: the fBM generator method from fbm_generators

        Output:
        diff_to_zero: numpy array, the absolute mean difference from zero for each sequence length

        Example:
        diff_zero(hurst=0.3,seq_range=[200,1200],seq_step=200,number_of_random_outcomes=10)
    """
    diff_to_zero = []

    for seq_len in range(*seq_range, seq_step):
        avg_zero = 0

        for _ in range(0, number_of_random_outcomes):
            X = fbm(hurst=hurst, n=seq_len, method=method)
            avg_zero += np.abs(X[1])

        diff_to_zero.append(avg_zero/number_of_random_outcomes)

    return diff_to_zero


Hursts = np.linspace(0.01, 0.99, 10)
seq_range = [100000, 1100000]
seq_step = 100000
number_of_random_outcomes = 1000

number_of_cores = multiprocessing.cpu_count()


def run(i):
    result = diff_zero(hurst=Hursts[i], seq_range=seq_range, seq_step=seq_step,
                       number_of_random_outcomes=number_of_random_outcomes)
    return result


if __name__ == '__main__':
    start = time.time()
    p = multiprocessing.Pool(number_of_cores)
    res = p.map(run, range(len(Hursts)))
    # print('output', res, len(res))
    plt.plot(res)
    plt.show()
    p.close()
    end = time.time()
    Mbox("The time of execution of above program is :", f"{end-start}", 1)
