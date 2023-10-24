import warnings
import numpy as np
import antropy as ant


def estimator(path, method="variogram", p=1, q=1):
    """
        Default estimator is the 1st order variogram (madogram).
        For variogram set p=2, for rodogram set p=1/2
        Methods can be variogram with different order, Higuchi and Gen. Hurst.

        Default q-moments for Gen.Hurst is q=1.
    """

    if method == "variogram":
        """
            n: sample length
            l: distance between 2 datapoints

            p the order of the variation
            p = 1 madogram
            p = 2 variogram
            p = 1/2 rodogram

            sample: fBm path
        """

        # rescaled_increments = np.array([np.linspace(start= 0 if i == 0 else sample[i-1], stop=sample[i], num=n) for i in range(len(sample) - 1)]).flatten()
        # summed_increments1 = np.sum([np.abs(rescaled_increments[i] - rescaled_increments[i-1])**p for i in range(len(rescaled_increments))])
        # summed_increments2 = np.sum([np.abs(rescaled_increments[i] - rescaled_increments[i-2])**p for i in range(len(rescaled_increments))])

        sum1 = np.sum([np.abs(path[i] - path[i-1]) **
                       p for i in range(len(path))])
        sum2 = np.sum([np.abs(path[i] - path[i-2]) **
                       p for i in range(len(path))])

        # V_p(l/n)
        def vp(increments, l):
            v_p = (1 / (2 * (len(path) - l))) * increments

            return v_p

            # D_Vp
        D = 2 - 1 / p * \
            ((np.log(vp(sum2, 2)) - np.log(vp(sum1, 1))) / np.log(2))

        # estimated Hurst
        hurst = 2 - D

        return hurst

    elif method == 'higuchi':

        higuchi_fractal_dim = ant.higuchi_fd(path)
        hurst = 2 - higuchi_fractal_dim

        return hurst

    elif method == 'gen-hurst':
        ###################################
        # https://github.com/PTRRupprecht/GenHurst/blob/master/genhurst.py
        # Calculates the generalized Hurst exponent H(q) from the scaling
        # of the renormalized q-moments of the distribution
        #
        #       <|x(t+r)-x(t)|^q>/<x(t)^q> ~ r^[qH(q)]
        #
        ####################################
        # H = genhurst(S,q)
        # S is 1xT data series (T>50 recommended)
        # calculates H, specifies the exponent q
        #
        # example:
        #   generalized Hurst exponent for a random vector
        #   H=genhurst(np.random.rand(10000,1),3)
        #
        ####################################
        # for the generalized Hurst exponent method please refer to:
        #
        #   T. Di Matteo et al. Physica A 324 (2003) 183-188
        #   T. Di Matteo et al. Journal of Banking & Finance 29 (2005) 827-851
        #   T. Di Matteo Quantitative Finance, 7 (2007) 21-36
        #
        ####################################
        ##   written in Matlab : Tomaso Aste, 30/01/2013 ##
        ##   translated to Python (3.6) : Peter Rupprecht, p.t.r.rupprecht (AT) gmail.com, 25/05/2017 ##

        # https://de.mathworks.com/matlabcentral/fileexchange/36487-weighted-generalized-hurst-exponent?s_tid=srchtitle
        # https://de.mathworks.com/matlabcentral/fileexchange/30076-generalized-hurst-exponent?s_tid=srchtitle

        L = len(path)
        if L < 100:
            warnings.warn('Data series very short!')

        H = np.zeros((len(range(5, 20)), 1))
        k = 0

        for Tmax in range(5, 20):

            x = np.arange(1, Tmax+1, 1)
            mcord = np.zeros((Tmax, 1))

            for tt in range(1, Tmax+1):
                dV = path[np.arange(tt, L, tt)] - \
                    path[np.arange(tt, L, tt)-tt]
                VV = path[np.arange(tt, L+tt, tt)-tt]
                N = len(dV) + 1
                X = np.arange(1, N+1, dtype=np.float64)
                Y = VV
                mx = np.sum(X)/N
                SSxx = np.sum(X**2) - N*mx**2
                my = np.sum(Y)/N
                SSxy = np.sum(np.multiply(X, Y)) - N*mx*my
                cc1 = SSxy/SSxx
                cc2 = my - cc1*mx
                ddVd = dV - cc1
                VVVd = VV - \
                    np.multiply(cc1, np.arange(1, N+1, dtype=np.float64)) - cc2
                mcord[tt-1] = np.mean(np.abs(ddVd)**q) / \
                    np.mean(np.abs(VVVd)**q)

            mx = np.mean(np.log10(x))
            SSxx = np.sum(np.log10(x)**2) - Tmax*mx**2
            my = np.mean(np.log10(mcord))
            SSxy = np.sum(np.multiply(
                np.log10(x), np.transpose(np.log10(mcord)))) - Tmax*mx*my
            H[k] = SSxy/SSxx
            k = k + 1

        mH = np.mean(H)/q

        return mH

    else:
        pass
