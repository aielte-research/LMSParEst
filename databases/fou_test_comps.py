import numpy as np
from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
import time
from fou import gen, fou_class
from utils import excep_raiser, cond
from fbm_generators import fbm

def method_setter(series_length, hurst, sigma, alpha, time_scale, initial_value, hurst_method,fou_method):
    params = locals()
    params['initial_value'] = 0.0
    excep_raiser(fou_method in ["integrated_formula","fbm_increment","ito_chaos_function","ito_chaos_class"],"Unknown generator method")
    if fou_method == "ito_chaos_class":
        generator = fou_class(**params)
        generator.get_item()
    else:
        if fou_method == "integrated_formula":
            params['initial_value'] = 1
        return gen(**params)

def exec_time_class_func_hurst(time_scale, length, alpha, sigma,random_outcomes,Hursts=[0.01,0.03,10],fbm_methods=["cholesky","daviesharte","hosking"],plot=False):
    """
    ---Description---
    Execution time comparison of the generator methods according to each Hurst exponent using the written functions for each given method and the firs_ito_class abstract class for the given parameters.
    
    ---Example---
    exec_time_class_func_hurst(time_scale=1.0,length=1000,alpha=1.0,sigma=0.5,random_outcomes=1,plot=True,Hursts=[0.001,0.999,20])
    """
    #excep_raiser(fbm_methods in ["cholesky","daviesharte","hosking"],"Unknown generator method")

    hursts = np.linspace(*Hursts)
    print("Applying class object as generator.")
    execution_time = np.zeros((len(fbm_methods)+1,len(hursts)))

    for hurst_idx, Hurst in enumerate(hursts):
        start_time = time.time()
        generator = fou_class(time_scale=time_scale,series_length=length,hurst=Hurst,alpha=alpha,sigma=sigma)
        for _ in range(random_outcomes):
            generator.get_item()
        execution_time[0,hurst_idx] = (time.time() - start_time) / random_outcomes

    print("Applying function object as generator.")
    for method_idx, method in enumerate(fbm_methods):
        print("The", method_idx+1, ". driving noise is generated accoding to: "+method)
        for hurst_idx, Hurst in enumerate(hursts):
            start_time = time.time()
            for _ in range(random_outcomes):
                gen(hurst=Hurst,series_length=length,alpha=alpha,sigma=sigma,fbm_method=method,ito_chaos_decomp=False)
            execution_time[method_idx+1,hurst_idx] = (time.time() - start_time) / random_outcomes

    if plot == True:
        plt.plot(hursts,execution_time[0,:],label="fou_class")
        for method_idx, method in enumerate(fbm_methods):
            plt.plot(hursts,execution_time[method_idx+1,:],label=method)
        plt.legend(title="Execution Time for Hurst Exponent: "+str(Hursts)+" and Serie Length: "+str(length))
        plt.rcParams["figure.figsize"] = (30,15)
        plt.xlabel("The Hurst Exponent")
        plt.ylabel("Execution time in sec")
        plt.show()

    return execution_time

def exec_time_class_func_len(time_scale, alpha, sigma,random_outcomes,hurst,lengths=[10000,100000,10000], fbm_methods=["cholesky","daviesharte","hosking"],plot=False):
    """
    ---Description---
    Execution time comparison of the generator methods according to each serie length using the written functions for each given method and the firs_ito_class abstract class for the given parameters.
    """
    lens = np.int_(np.linspace(*lengths))
    print("Applying class object as generator.")
    execution_time = np.zeros((len(fbm_methods)+1,len(lens)))

    for length_idx, Length in enumerate(lens):
        start_time = time.time()
        generator = fou_class(time_scale=time_scale,series_length=Length,hurst=hurst,alpha=alpha,sigma=sigma)
        for _ in range(random_outcomes):
            generator.get_item()
        execution_time[0,length_idx] = (time.time() - start_time) / random_outcomes

    print("Applying function object as generator.")
    for method_idx, method in enumerate(fbm_methods):
        print("The", method_idx+1, ". driving noise is generated accoding to: "+method)
        for length_idx, Length in enumerate(lens):
            start_time = time.time()
            for _ in range(random_outcomes):
                gen(hurst=hurst,series_length=Length,alpha=alpha,sigma=sigma,fbm_method=method,ito_chaos_decomp=False)
            execution_time[method_idx+1,length_idx] = (time.time() - start_time) / random_outcomes

    if plot == True:
        plt.plot(lens,execution_time[0,:],label="fou_class")
        for method_idx, method in enumerate(fbm_methods):
            plt.plot(lens,execution_time[method_idx+1,:],label=method)
        plt.legend(title="Execution Time for Hurst Exponent: "+str(hurst)+" and Serie Length: "+str(lens))
        plt.rcParams["figure.figsize"] = (30,15)
        plt.xlabel("The Serie Length")
        plt.ylabel("Execution time in sec")
        plt.show()

    return execution_time

