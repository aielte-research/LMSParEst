"""
---Description---
Generator system for the first Wiener-Ito chaos using Fast Fourier Transform as the fastest method to create fractional noise.  
SUGGESTION: If one aims at generating many sequences independently with the same parameters, then use the code including class object.
The generator procedure is much faster in this case.

@authors: Ivan Ivkovic
@contact: ivkovic@cs.elte.hu
@copyright: Ivan Ivkovic, Daniel Janos Boros
@deprecated: False
@date: 2021.11.13
@version: v.0.1.2
"""
# Debugged: get_fbm_conv, cov_strucure_embedding, get_fbm_increments, first_ito_chaos_gen, first_ito_chaos_class.embedded_cov_structure
import numpy as np
import cmath
from scipy.fftpack import fft
from databases.fbm_generators import fbm
from databases.utils import in_interval, excep_raiser, cond_exception

get_fbm_cov = lambda length, hurst: [1.0] + [1/2*((idx+1)**(2*hurst)-2*idx**(2*hurst)+(idx-1)**(2*hurst)) for idx in range(1,length+1)]
"""
    ---Description---
    get_fbm_cov returns the covariance structure of a fBM according to the given parameters

    ---Inputs---
    @arg length: the length of the discrete fBM, int
    @arg hurst: Hurst exponent, float from [0.0,1.0]

    ---Outputs---
    the auto-covariance structure of the fbm with the given attributes, np.array with (length+1) length
    """

cov_structure_embedding = lambda cov_structure: [cmath.sqrt(x) for x in np.real(fft(np.concatenate([cov_structure, cov_structure[::-1][1:len(cov_structure)-1]]))/(2*(len(cov_structure)-1)))]

def get_fbm_increments(cov_structure,hurst):
    # TASK: speed up the embedding
    """
    ---Description---
    get_fbm_increments returns fractional Wiener increments with the given covariance structure and Hurst exponent.

    ---Inputs---
    @arg cov_structure: the discrete auto-covariance function of a fBM, one can obtain it by applying get_fbm_cov with the corresponding attributes, np.array
    @arg hurst: Hurst exponent, float from [0.0,1.0]

    ---Outputs---
    len(cov_structure)-1 fBM increments
    """

    embedded_cov_mtx = cov_structure_embedding(cov_structure=cov_structure)
    W = fft(embedded_cov_mtx*(np.random.normal(size=2*(len(cov_structure)-1))+np.random.normal(size=2*(len(cov_structure)-1))*complex(0, 1)))
    return  (len(cov_structure)-1)**(-hurst)*np.real(W[1:(len(cov_structure))])

def first_ito_chaos_gen(hurst,time_scale,length,weights,fbm_gen=False):
    """
    ---Description---
    It returns an element of the first Wiener-Ito chaos according to the given discretized integrand.
    SUGGESTION: If one aims to generate many sequences independently with the same parameters, then use the class object.
                It is much faster in this case.

    ---Inputs---
    @arg hurst: Hurst exponent, float from [0.0,1.0]
    @arg time_scale: the time interval [0,time_scale] the process will be generated according to, positive float
    @arg length: the length of the discrete fBM, int
    @arg weights: The discretized integrand over the obtained grid by the parameters time_scale and length, 1D np.array
    @arg fbm_gen: Boolean variable for applying fbm() function in case of scaled fBM

    ---Auxiliary functions---
    get_fbm_cov: to get the covariance structure of the driving noise to generate the increments by fft method
    get_fbm_increments: to get the driving noise for the discretized Wiener-integral

    ---Outputs---
    An element from the first Wiener-Ito chaos according to the given parameters and discretized integrand, np.array

    ---Example---
    For an example check the fOU generator.
    """
    excep_raiser(in_interval([0.0,1.0],hurst),"The Hurst exponent has to fall into the interval [0,1]")
    excep_raiser(length==len(weights),"You have to declare exactly as many weights as the length of the time serie.")

    if np.all(weights==weights[0]) and fbm_gen:
        print("You actually aim at generating a scaled fBM. Please check the concrete fBM generator using fft method. It is much faster in this case.")
        X = weights[0]*fbm(hurst=hurst,n=length,length=time_scale,method="fft")
    else:
        cov_structure = get_fbm_cov(length-1,hurst)
        fbm_increments = get_fbm_increments(cov_structure,hurst)
        # let me consider the R-S integral according to the "left points" in each grid
        X = time_scale**hurst*np.cumsum(np.multiply([0,*fbm_increments],weights))
    return X



class first_ito_chaos_class():
    #TASK: Add caching --- https://towardsdatascience.com/how-to-speed-up-your-python-code-with-caching-c1ea979d0276
    """
    ---Description---
    Class object for efficiently generating elements of the first Wiener-Ito chaos by the get_item() method. Note that since the driving noise is a
    discrete stationary process, the embedding of the covariance structure can be much faster according to applying Toeplitz matrices in a tricky way. Moreover, 
    except the actual increment generating procedure every needed data or information can be cached, which gives a huge boost for the execution time.

    ---Initialization---
    @arg hurst: Hurst exponent, float from [0.0,1.0]
    @arg time_scale: the time interval [0,time_scale] the process will be generated according to, positive float
    @arg length: the length of the discrete fBM, int
    @arg weights: the discretized integrand over the obtained grid by the parameters time_scale and length, 1D np.array
    @arg embedded_cov_structure: the embedded covariance structure needed for generating increments, np.array
    
    ---Subroutines---
    __init__ : the defeault class initialization function
    get_item : it returns a descretized element of the first Wiener-Ito subspace according to the given parameters
    """
    def __init__(self,hurst,time_scale,length,weights):
        self.hurst = cond_exception(in_interval([0.0,1.0],hurst),hurst,'The Hurst exponent has to fall into the interval [0,1].')
        self.time_scale = time_scale
        self.length = length-1
        self.weights = weights
        self.embedded_cov_structure = cov_structure_embedding(get_fbm_cov(self.length,self.hurst))

    #get_item = lambda self: self.time_scale**self.hurst*np.cumsum(np.multiply([0,(self.length)**(-self.hurst)*np.real(W[1:(self.length+1)])],self.weights))
    def get_item(self): # goal: implement it as to obtain a much faster function
        W = fft(self.embedded_cov_structure*(np.random.normal(size=2*(self.length))+np.random.normal(size=2*(self.length))*complex(0, 1)))
        fbm_increments = (self.length)**(-self.hurst)*np.real(W[1:(self.length+1)])
        return self.time_scale**self.hurst*np.cumsum(np.multiply([0,*fbm_increments],self.weights))
    """
    ---Description---
    It returns an element of the first Wiener-Ito chaos according to the given discretized integrand.
    SUGGESTION: If one aims to generate many sequences independently with the same parameters, then use the class object.
                It is much faster in this case.

    ---Inputs---
    @arg hurst: Hurst exponent, float from [0.0,1.0]
    @arg time_scale: the time interval [0,time_scale] the process will be generated according to, positive float
    @arg length: the length of the discrete fBM, int
    @arg weights: The discretized integrand over the obtained grid by the parameters time_scale and length, 1D np.array

    ---Outputs---
    An element from the first Wiener-Ito chaos according to the given parameters and discretized integrand, np.array

    ---Example---
    For an example check the fOU generator.
    """
    


