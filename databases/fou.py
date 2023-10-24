"""
---Description---
Fractional Ornstein-Uhlenbeck with several options, e.g. using the integrated formula, using discretized Riemann-Stieltjes integral in
case of special initial value and using the first_ito_chaos_gen as the fastest method using fft also in case of special initial value.
SUGGESTION: If one aims to generate many sequences independently with the same parameters by applying first_ito_chaos_gen, 
then use the code including class object. It is much faster in this case.

@authors: Ivan Ivkovic
@contact: ivkovic@cs.elte.hu
@copyright: Ivan Ivkovic, Daniel Janos Boros
@deprecated: False
@date: 2021.11.13
@version: v.0.1.5
"""

import numpy as np
from fbm_generators import fbm
from first_ito_chaos import first_ito_chaos_gen, first_ito_chaos_class, cov_structure_embedding, get_fbm_cov, get_fbm_increments
from utils import cond_exception, in_interval, cond
from scipy.fftpack import fft

drift = lambda drift_param, prev_value, time_step: -drift_param*prev_value*time_step
diffusion = lambda volatility, noise_increment: volatility*noise_increment
exp_incremet = lambda sigma, alpha, time_scale, observation_time, noise_increment: sigma*np.exp(-alpha*(time_scale-observation_time))*noise_increment

fou_discrete_integrand = lambda timescale, length, sigma, alpha: sigma*np.exp(-alpha*(timescale*np.ones(length)-[np.linspace(0,timescale,length)]))
"""
---Description---
Returns the discretized deterministic fractional Ornstein-Uhlenbeck integrand according to the given parameters.

---Inputs---
@arg length: the length of the sequence, int
@arg sigma: the diffusion parameter, positive float
@arg alpha: the drift parameter, positive float
@arg time_scale: the time interval [0,time_scale] the fOU process will be generated according to, positive float
---Output--

---Example---
Check the fou_class function
"""

def gen(series_length,hurst,sigma,alpha,time_scale=1.0,initial_value=0.0,fbm_method="fft",ito_chaos_decomp=True):
# TASK: mean reversion
    """
    ---Description---
    Generator for fractional Orntsein-Uhlenbeck process

    ---Inputs---
    @arg series_length: the length of the sequence, int
    @arg hurst: the Hurst exponent: [0,1]
    @arg sigma: the diffusion parameter, positive float
    @arg alpha: the drift parameter, positive float
    @arg time_scale: the time interval [0,time_scale] the fOU process will be generated according to, positive float
    @arg initial_value: the initial value, float
    @arg fbm_method: the method the driving noise has been generated according to, string from <"cholesky", "daviesharte", "hosking", "fft">
    @arg ito_chaos_decomp: it offers a fast generator option in case of special initial value by 
                           using the first Wiener-Ito chaos generator to obtain a fOU serie according to the given parameters, boolean

    ---Auxiliary functions---
    drift: the drift increment of the fOU process
    diffusion: the diffusion increment of the fOU process
    exp_increment: the exponential increment of the fOU process written in Wiener-Ito chaos form in chase of 0 initial value
    fou_discrete_integrand: dicretized version of the fOU integrand, this will be dot producted with the fractional noise

    ---Output---
    A discretized realization of a fractional Ornstein-Uhlenbeck trajectory according to the input parameters

    ---Example---
    TASK till the last version
    """

    if initial_value == 0:
        """
        In case of sepcified initial value one can use the formula obtained by the fact that fOU processes with
        0 initial value are elements of the first Wiener-Ito chaos according to the driving fBM. The method now
        consists of just a dicretization of a pathwise Riemann-Stieltjes integral.
        """

        if ito_chaos_decomp == False:
            """
            In this case the fact, that with 0 initial value dicrete fOU can be written as a discrete Riemann-Stieltjes sum, lets us
            obtain the fOU serie as a weighted sum of the fBM increments according to the fft method.
            """
            Fbm = fbm(hurst=hurst,n=series_length-1,method=fbm_method,length=time_scale)

            dt = (time_scale)/(series_length-1)
            
            X = np.zeros(series_length)

            for i in range(1,series_length):
                X[i] = X[i-1] + exp_incremet(sigma=sigma,alpha=alpha,time_scale=time_scale,observation_time=i*dt,noise_increment=Fbm[i]-Fbm[i-1])

        else:
            """
            In this case we apply the first Wiener-Ito chaos generator for the fOU deterministic integrand, check fou_discrete_integrand.
            That is the fastest way to generate that kind of process.
            """
            X = first_ito_chaos_gen(hurst=hurst,time_scale=time_scale,length=series_length,weights=fou_discrete_integrand(time_scale,series_length,sigma,alpha))

    else:
        """
        The standard generator method according to the integrated formula of the fOU process.
        """
        if fbm_method == "fft":
            Fbm = fbm(hurst=hurst,n=series_length-1,method=fbm_method,length=time_scale)
        else:
            Fbm = fbm(hurst=hurst,n=series_length,method=fbm_method,length=time_scale)

        dt = (time_scale)/(series_length-1)

        X = np.zeros(series_length) 
        X[0] = initial_value
        for i in range(1,series_length):
            X[i]=(X[i-1]+drift(drift_param=alpha,prev_value=X[i-1],time_step=dt)
                +diffusion(volatility=sigma,noise_increment=Fbm[i]-Fbm[i-1]))

    return X


class general_fou_class():
    """
    ---Description---
    Efficient generator object for fOU process with arbitrary initial value, suggested for non-zero init value

    ---Inputs---
    @arg series_length: the length of the sequence, int 
    (series_length data points will be simulated, which means for a given series_length and initial_value there will be series_length+1 data points returned)
    @arg hurst: the Hurst exponent: [0,1]
    @arg sigma: the time-dependent diffusion parameter, positive float array or number
    @arg alpha: the drift parameter, positive float
    @arg mu: the mean-reverting parameter, float
    @arg time_scale: the time interval [0,time_scale] the fOU process will be generated according to, positive float
    @arg initial_value: the initial value, float

    ---Subroutines---
    get_fbm_cov: returns the covariance structure of the fractional Wiener process over a given grid
    embedded_cov_structure: returns a structure (matrix) obtained by applying circulant matrix embedding

    ---Output---
    A discretized realization of a fractional Ornstein-Uhlenbeck trajectory according to the input parameters

    ---Example---
    generator_object = general_fou_class(series_length=1500,hurst=0.7,sigma=1.2,alpha=0.9,time_scale=1.0,initial_value=2.6)
    generator_object.get_item()
    """
    def __init__(self,series_length,hurst,sigma,alpha,mu,time_scale,initial_value):
        # Parameters
        self.hurst = cond_exception(in_interval([0.0,1.0],hurst),hurst,'The Hurst exponent has to fall into the interval [0,1].')
        self.time_scale = time_scale
        self.length = series_length
        self.alpha = cond_exception(alpha>0,alpha,'This parameter has to be positive.')
        self.sigma = cond_exception(sigma>0,sigma,'This parameter has to be positive.')
        self.mu = mu
        self.initial_value = initial_value
        self.time_step = time_scale/(series_length-1)
        # Subroutine which actually need to be calculated just once
        self.embedded_cov_structure = cov_structure_embedding(get_fbm_cov(self.length,self.hurst))

    def get_item(self):
        """
        Simulating one sequence according to the given attributes
        """
        W = fft(self.embedded_cov_structure*(np.random.normal(size=2*(self.length))+np.random.normal(size=2*(self.length))*complex(0, 1)))
        fbm_increments = (self.length)**(-self.hurst)*np.real(W[1:(self.length+1)])

        # let the first auxiliary increment be the non-zero initial value
        fou_increments = np.zeros(self.length+1)
        fou_increments[0] = self.initial_value

        # the following array will be returned with length series_length+1
        simulated_process = np.zeros(self.length+1)
        simulated_process[0] = self.initial_value

        for idx in range(1,self.length+1):
            fou_increments[idx] = self.alpha * self.time_step *(self.mu - simulated_process[idx-1]) + self.sigma * fbm_increments[idx-1]
            simulated_process[idx] = fou_increments[idx] + simulated_process[idx-1]

        return simulated_process


def fou_class(series_length,hurst,sigma,alpha,time_scale,initial_value=0):
    """
    ---Description---
    The fastest way to generate fractional Ornstein-Uhlenbeck processes with arbitrary initial value by applying the class object written in first_ito_chaos.py according to
    the discretized integrand of fOU (fou_discrete_integrand)for zero initial value or in case of arbitrary chosen initial value by applying the general_fou_class.

    ---Input---
    @arg length: the length of the sequence, int
    @arg hurst: the Hurst exponent: [0,1]
    @arg sigma: the diffusion parameter, positive float
    @arg alpha: the drift parameter, positive float
    @arg time_scale: the time interval [0,time_scale] the fOU process will be generated according to, positive float
    @arg initial_value: the initial value of the corresponding stochastic differential equation, float

    ---Subroutines---
    fou_discrete_integrand: returns the discretized deterministic fOU integrand according to the given parameters

    ---Output---
    Regarding the initial value:
    A general_fou_class object defined above or a firs_ito_chaos_class object from first_ito_fbm.py with weights obtained by fou_discrete_integrand() function. 
    -> One can generate independent sequences by calling class method .get_item().

    ---Example---
    X = fou_class(300,0.3,0.5,1.0,1.0)
    X.get_item()
    """
    if initial_value == 0:
        return (first_ito_chaos_class(hurst=hurst,
                                    time_scale=time_scale,
                                    length=series_length,
                                    weights=fou_discrete_integrand(time_scale,
                                                                    series_length,
                                                                    cond_exception(sigma>0,sigma,'This parameter has to be positive.'),
                                                                    cond_exception(alpha>0,alpha,'This parameter has to be positive.'))))
    else:
        return (general_fou_class(series_length=series_length,hurst=hurst,
                sigma=sigma,alpha=alpha,time_scale=time_scale,initial_value=initial_value))                                                            

class fOU():
    def __init__(self,series_length,hurst,sigma,alpha,mu,initial_value,grid,time_scale=1,equidistant_grid=True):
        # Parameters
        self.hurst = cond_exception(in_interval([0.0,1.0],hurst),hurst,'The Hurst exponent has to fall into the interval [0,1].')
        self.time_scale = time_scale
        self.length = series_length
        self.alpha = cond_exception(alpha>0,alpha,'This parameter has to be positive.')
        self.sigma = cond_exception(sigma>0,sigma,'This parameter has to be positive.')
        self.mu = mu
        self.initial_value = initial_value
        self.grid = cond(not equidistant_grid, grid, [i * time_scale/(series_length-1) for i in range(0,series_length+1)])
        # Subroutine which actually need to be calculated just once
        # self.embedded_cov_structure = cov_structure_embedding(get_fbm_cov(self.length,self.hurst))

    # get_fbm_cov = lambda self: [1.0] + [1/2*((idx+1)**(2*self.hurst)-2*idx**(2*self.hurst)+(idx-1)**(2*self.hurst)) for idx in range(1,self.length+1)]

    def embedded_cov_structure(self):
        pass

    def get_item(self):
        pass