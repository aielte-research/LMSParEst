"""
---Description---
Containing several tests and comparisons about the generator methods according to execution time and accuracy obtained by the
estimation methods from fbm_hurst_estimators.py

@authors: Ivan Ivkovic
@contact: ivkovic@cs.elte.hu
@copyright: Ivan Ivkovic, Daniel Janos Boros
@deprecated: False
@date: 2021.11.13
@version: v.0.1.0
"""

import numpy as np
from fbm_generators import fbm
from fbm_hurst_estimators import estimator
from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
import time



def plot_path_fbm(hurst,time_scale,length,methods=["cholesky","daviesharte","hosking","fft"]):
    """
    ---Description---
    A plot function which returns fBM paths obtained by the given methods.

    ---Inputs---
    @arg hurst: the Hurst exponent, float from [0,1]
    @arg time_scale: the interval the process will be generated according to, [0.0,time_scale], float
    @arg length: the length of the time serie, int
    @arg methods: the fBM generators from fbm_generators.py, ["cholesky","daviesharte","hosking","fft"]

    ---Output---
    Plot of the generated paths for each method.


    ---Example---
    plot_path_fbm(hurst=np.linspace(0.001,0.99,5),length=1000,time_scale=1.0)
    """

    for method_idx in range(len(methods)):
        print("The", method_idx+1, ". method is: "+methods[method_idx])
        Fbms = [fbm(hurst=H,n=length,method=methods[method_idx],length=time_scale) for H in hurst]
        for hurst_idx in range(len(hurst)):
            string = "Hurst: " + str(hurst[hurst_idx])
            plt.plot(Fbms[hurst_idx],label=string)
        plt.legend(title=methods[method_idx])
        plt.rcParams["figure.figsize"] = (30,15)
        plt.show()



def execution_time_fbm_lens(lengths,random_outcomes,hurst,time_scale,methods=["cholesky","daviesharte","hosking","fft"],plot=False):
    """
    ---Description---
    The function returns an average execution time of the generating process according to some fixed series lengths with given
    Hurst exponent, time scale parameter and generator method. The applied methods can be found in fbm_generators.py. An additional
    option is obtainig a figure from the plots of the average execution times for each given generator methods.

    ---Inputs---
    @arg hurst: the Hurst exponent, float from [0,1]
    @arg time_scale: the interval the process will be generated according to, [0.0,time_scale], float
    @arg lengths: the lengths of the time series, int array: [lengths_1,lengths_2,lengths_3] -> the lengths: np.linspace(*lengths)
    @arg methods: the fBM generator methods from fbm_generators.py, ["cholesky","daviesharte","hosking","fft"]
    @arg random_outcomes: the number of random outcomes the execution time will be averaged over, int
    @arg plot: plot the results, boolean

    ---Output---
    A two-diemnsional float np.array, with dimensions len(methods)xlen(np.linspace(*lengths)), containing the average execution time 
    for each given method-series length pair.
    Plot of the execution time depending on the time serie length for each method. (Optional depending on the boolean variable plot.)

    ---Example---
    execution_time_fbm_lens(lengths=[100000,2000000,10],random_outcomes=100,hurst=0.7,time_scale=1.0,plot=True)
    """
    lens = np.linspace(*lengths)
    execution_time = np.zeros((len(methods),len(lens)))

    for method_idx in range(len(methods)):
        print("The", method_idx+1, ". method is: "+methods[method_idx])
        for idx in range(len(lens)):
            for _ in range(random_outcomes):
                N = int(lens[idx])
                start_time = time.time()
                fbm(hurst=hurst,n=N,length=time_scale,method=methods[method_idx])
                execution_time[method_idx,idx] += time.time() - start_time

    if plot == True:
        for method_idx in range(len(methods)):
            plt.plot(lens,execution_time[method_idx,:],label=methods[method_idx])
        plt.legend(title="Execution Time for Hurst Exponent: "+str(hurst))
        plt.rcParams["figure.figsize"] = (30,15)
        plt.xlabel("Length of The Time Serie")
        plt.ylabel("Execution time in sec")
        plt.show()


    return execution_time/random_outcomes


def execution_time_fbm_hursts(length,random_outcomes,hurst,time_scale,methods=["cholesky","daviesharte","hosking","fft"],plot=False):
    """
    ---Description---
    The function returns an average execution time of the generating process according to some fixed Hurst exponents with given
    series length, time scale parameter and generator method. The applied methods can be found in fbm_generators.py. An additional
    option is obtainig a figure from the plots of the average execution times for each given generator methods.


    ---Inputs---
    @arg hurst: the Hurst exponent, float array from [0,1]: [hurst_1,hurst_2,hurst_3] -> the Hurst exponents: np.linspace(*hurst)
    @arg time_scale: the interval the process will be generated according to, [0.0,time_scale], float
    @arg length: the length of the time series, int
    @arg methods: the fBM generator methods from fbm_generators.py, ["cholesky","daviesharte","hosking","fft"]
    @arg random_outcomes: the number of random outcomes the execution time will be averaged over, int
    @arg plot: plot the results, boolean

    ---Output---
    A two-diemnsional float np.array, with dimensions len(methods)xlen(np.linspace(*hurst)), containing the average execution time 
    for each given method-Hurst exponent pair.
    Plot of the execution time depending on the Hurst exponent for each method. (Optional depending on the boolean variable plot.)

    ---Example---
    execution_time_fbm_hursts(hurst=[0.1,0.7,10],random_outcomes=100,length=10000,time_scale=1.0,plot=True)
    """
    hursts = np.linspace(*hurst)
    execution_time = np.zeros((len(methods),len(hursts)))

    for method_idx in range(len(methods)):
        print("The", method_idx+1, ". method is: "+methods[method_idx])
        for idx in range(len(hursts)):
            for _ in range(random_outcomes):
                H = hursts[idx]
                start_time = time.time()
                fbm(hurst=H,n=length,length=time_scale,method=methods[method_idx])
                execution_time[method_idx,idx] += time.time() - start_time

    if plot == True:
        for method_idx in range(len(methods)):
            plt.plot(hursts,execution_time[method_idx,:],label=methods[method_idx])
        plt.legend(title="Execution Time for Hurst Exponent: "+str(hurst))
        plt.rcParams["figure.figsize"] = (30,15)
        plt.xlabel("The Hurst Exponent")
        plt.ylabel("Execution time in sec")
        plt.show()


    return execution_time/random_outcomes
