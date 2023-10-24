import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr

from databases.fbm_gen import base_gen as fbm
from databases.fou_gen import general_fou_class


class Heston:
    """
    @authors: Dalma Toth-Lakits, Daniel Boros
    @contact:
    @copyright: Dalma Toth-Lakits, Daniel Boros
    @deprecated: False
    @date: 2022.03.02
    @version: v.0.0.1
    ---Description---
        Modelling the discretized Heston model with different approaches, like constant correlation or non-constant correlation and furthermore
        there is an option for fractional Heston model too.
    ---Inputs---
        @arg n: length of the sequence
        @arg S0: float initial assets' price
        @arg v0: float
        @arg mu1: float param of price process
        @arg kappa: float param of vol process
        @arg theta: float param of vol process
        @arg eta: float param of vol process
        @arg alpha: float param of correlation process
        @arg mu2: float param of correlation process
        @arg sigma: float param of correlation process
        @arg fou_hurst: float param of correlation process
        @arg hurst1: float param of fBm1
        @arg hurst2: float param of fBm2
        @arg T: float
        @arg dt: float
        @arg rho: float
        @arg const_correlation: bool (default=False)
        @arg rough_correlation: bool (default=False)
        @arg fractional_wieners: bool (default=False)
        @arg hybrid_wieners: bool (default=False)
        @arg rough_volatility: bool (default=True)
    ---Output---
        S: price process
        v: volatility process
        rho: correlation process
        w1, w2: correlated Wieners
    ---Example---
        heston = Heston()
        heston.model()
        For plotting the correlated Wieners set the plot_wiernes=True
        heston.plot(plot_wieners=True)
        Rough model
        heston = Heston()
        heston.rough_model()
    """

    def __init__(self, n=1000,
                 S0=0.05, v0=0.03, kappa=0.07, alpha=0.06,
                 mu1=0.01, mu2=0.5, sigma=0.05, theta=0.05, eta=0.03,
                 T=20, dt=0.001, rho=-0.3, fou_hurst=0.02,
                 hurst1=0.7, hurst2=0.7,
                 const_correlation=True, rough_correlation=False, fractional_wieners=True, hybrid_wieners=False, rough_volatility=False,
                 discretization='euler'):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.alpha = alpha
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma
        self.theta = theta
        self.eta = eta
        self.T = T
        self.dt = dt
        self.fou_hurst = fou_hurst
        self.hurst1 = hurst1
        self.hurst2 = hurst2
        self.discretization = discretization

        # set constant correlation
        self.const_correlation = const_correlation
        # set rough correlation
        self.rough_correlation = rough_correlation
        # correlated Wieners are fractional processes
        self.fractional_wieners = fractional_wieners
        # correlated Wieners are ordinary and fractional processes
        self.hybrid_wieners = hybrid_wieners
        # in hybrid model the volatility process is the default fractional process
        # set for False to swap with the price process
        # it works only if the hybrid_wieners=True
        self.rough_volatility = rough_volatility

        self.n = n
        self.t = np.linspace(0, T, self.n)
        self.sqrtdt = np.sqrt(dt)
        self.w = np.zeros(n)
        self.w1 = np.zeros(n)
        self.w2 = np.zeros(n)
        self.dw = np.zeros(n)
        self.dw1 = np.zeros(n)
        self.dw2 = np.zeros(n)
        self.s = np.zeros(n)
        self.ds = np.zeros(n)
        self.S = np.zeros(n)
        self.dS = np.zeros(n)
        self.v = np.zeros(n)
        self.dv = np.zeros(n)
        self.X = np.zeros(n)
        self.dX = np.zeros(n)
        self.rho = np.full(n, rho)
        self.square_root_v = np.zeros(n)
        self.square_root_dv = np.zeros(n)

    def model(self):
        dS = self.dS
        S = self.S
        dv = self.dv
        v = self.v

        S[0] = self.S0
        v[0] = self.v0

        if not self.rough_correlation:
            self.correlation_process()
        else:
            self.rough_correlation_process()

        if not self.fractional_wieners and not self.hybrid_wieners:
            self.correlated_wieners()
        elif self.fractional_wieners:
            self.correlated_fractional_wieners()
        else:
            self.hybrid_wieners()

        for i in range(self.n-1):
            if self.discretization == 'euler':
                dv[i] = self.kappa*(self.theta-v[i])*self.dt + \
                    self.eta*np.sqrt(v[i])*self.dw2[i]
                v[i+1] = max(v[i]+dv[i], 0.00001)
            elif self.discretization == 'ito-transformed':
                dv[i] = 1 / \
                    (2*np.sqrt(max(v[i], 0.00001))) * (self.kappa*self.theta-self.kappa *
                                                       dv[i]-1/4*self.eta**2)*self.dt+1/2*self.eta*np.sqrt(self.dt)*self.dw2[i]
                v[i+1] = np.sqrt(max(v[i] +
                                     dv[i], 0.00001))
            elif self.discretization == 'khi-square':
                df = 4*self.kappa*self.theta / self.eta**2
                nc = (4*self.kappa*np.exp(-self.kappa*self.dt)) / \
                    self.eta**2*(1-np.exp(-self.kappa*self.dt))*v[i]

                v[i+1] = (self.eta**2*(1-np.exp(-self.kappa*self.dt)) /
                          4*self.kappa)*np.random.noncentral_chisquare(df, nc)

            dS[i] = self.mu1*S[i]*self.dt+np.sqrt(v[i])*S[i]*self.dw1[i]
            S[i+1] = S[i]+dS[i]

        self.S = S
        self.v = v

        return S, v, #self.rho, self.dw1, self.dw2

    def correlated_wieners(self):
        ds = self.ds
        dw1 = self.dw1
        dw2 = self.dw2
        w1 = self.w1
        w2 = self.w2

        for i in range(self.n-1):
            dw1[i] = self.sqrtdt * np.random.randn()
            ds[i] = self.sqrtdt * np.random.randn()
            dw2[i] = self.rho[i] * dw1[i] + \
                np.sqrt(1 - self.rho[i] ** 2) * ds[i]
            w1[i + 1] = w1[i] + dw1[i]
            w2[i + 1] = w2[i] + dw2[i]

        self.ds = ds
        self.dw1 = dw1
        self.dw2 = dw2
        self.w1 = w1
        self.w2 = w2

    def correlated_hydrid_wieners(self):
        w1 = fbm(hurst=self.hurst1, n=self.n, length=self.T)
        dw1 = np.diff(w1)
        ds = self.ds
        dw2 = self.dw2
        s = self.s
        w2 = self.w2

        for i in range(self.n-1):
            ds[i] = self.sqrtdt * np.random.randn()
            dw2[i] = self.rho[i] * dw1[i] + \
                np.sqrt(1 - self.rho[i] ** 2) * ds[i]
            s[i + 1] = s[i] + ds[i]
            w2[i + 1] = w2[i] + dw2[i]

        if self.rough_volatility:
            self.ds = ds
            self.dw1 = dw2
            self.dw2 = dw1
            self.s = s
            self.w1 = w2
            self.w2 = w1
        else:
            self.ds = ds
            self.dw1 = dw1
            self.dw2 = dw2
            self.s = s
            self.w1 = w1
            self.w2 = w2

    def correlated_fractional_wieners(self):
        s = fbm(hurst=self.hurst1, n=self.n, T=self.T)
        ds = np.diff(s)
        w1 = fbm(hurst=self.hurst2, n=self.n, T=self.T)
        dw1 = np.diff(w1)
        dw2 = self.dw2
        w2 = self.w2

        for i in range(self.n-1):
            dw2[i] = self.rho[i] * dw1[i] + \
                np.sqrt(1 - self.rho[i] ** 2) * ds[i]
            w2[i + 1] = w2[i] + dw2[i]

        self.ds = ds
        self.dw1 = dw1
        self.dw2 = dw2
        self.s = s
        self.w1 = w1
        self.w2 = w2

    def correlation_process(self):
        if self.alpha < 0:
            raise ValueError('alpha must be a positive value!')

        dw = self.dw
        dX = self.dX
        X = self.X
        rho = self.rho

        if not self.const_correlation:
            for i in range(self.n-1):
                dw[i] = self.sqrtdt * np.random.randn()
                dX[i] = self.alpha*(self.mu2-X[i])*self.dt+self.sigma*dw[i]
                X[i+1] = X[i]+dX[i]
                rho[i+1] = np.tanh(X[i+1])

            self.dw = dw
            self.dX = dX
            self.X = X
            self.rho = rho
        else:
            self.rho = np.full(self.n, rho)

    def rough_correlation_process(self):
        if self.alpha < 0:
            excep_raiser('alpha must be a positive value!')

        fou = general_fou_class(series_length=self.n-1, hurst=self.fou_hurst,
                                sigma=self.sigma, alpha=self.alpha, mu=self.mu2, time_scale=self.T, initial_value=0)

        fou_path = fou.get_item()

        self.rho = np.tanh(fou_path)

    @staticmethod
    def createChunks(data, split_size, batch_size, iterations):
        chunked_data = np.zeros((iterations, split_size, batch_size))

        for i in range(len(data)):
            # no overlap
            for j in range(0, len(data[i]), batch_size):
                for k in range(batch_size):
                    chunked_data[i][j //
                                    batch_size][k] = data[i][j:j+batch_size][k]

        return chunked_data

    def correlation_fitting(self, iterations, batch_size, show_plot=True):
        """
            @arg iterations: int (num of iterations, for fit)
            @arg batch_size: int (size of the disjoint windows)
            @arg show_plot: boolean (default=True)
        """
        split_size = self.n//batch_size

        S_array = np.zeros((iterations, self.n))
        v_array = np.zeros((iterations, self.n))
        rho_array = np.zeros((iterations, self.n))
        dw1_array = np.zeros((iterations, self.n))
        dw2_array = np.zeros((iterations, self.n))

        for j in range(iterations):
            S, v, rho, dw1, dw2 = self.model()
            S_array[j] = S
            v_array[j] = v
            rho_array[j] = rho
            dw1_array[j] = dw1
            dw2_array[j] = dw2

        batched_S = self.createChunks(
            data=S_array, split_size=split_size, batch_size=batch_size, iterations=iterations)
        batched_v = self.createChunks(
            data=v_array, split_size=split_size, batch_size=batch_size, iterations=iterations)
        batched_rho = self.createChunks(
            data=rho_array, split_size=split_size, batch_size=batch_size, iterations=iterations)
        batched_dw1 = self.createChunks(
            data=dw1_array, split_size=split_size, batch_size=batch_size, iterations=iterations)
        batched_dw2 = self.createChunks(
            data=dw2_array, split_size=split_size, batch_size=batch_size, iterations=iterations)

        sv_batched_corr = np.zeros((iterations, split_size))
        wieners_batched_corr = np.zeros((iterations, split_size))
        rho_batched_mean = np.zeros((iterations, split_size))

        for k in range(iterations):
            for l in range(split_size):
                wieners_batched_corr[k][l] = pearsonr(
                    batched_dw1[k][l], batched_dw2[k][l])[0]
                sv_batched_corr[k][l] = pearsonr(
                    batched_S[k][l], batched_v[k][l])[0]
                rho_batched_mean[k][l] = np.mean(batched_rho[k][l])

        sv_mean_batch_size = np.zeros(iterations)
        wieners_mean_batch_size = np.zeros(iterations)
        rho_mean_batch_size = np.zeros(iterations)

        for h in range(iterations):
            sv_mean_batch_size[h] = np.mean(sv_batched_corr[h])
            wieners_mean_batch_size[h] = np.mean(wieners_batched_corr[h])
            rho_mean_batch_size[h] = np.mean(rho_batched_mean[h])

        abs_error = np.abs(
            (wieners_mean_batch_size - rho_mean_batch_size)) * 100

        abs_error_mean = np.mean(abs_error)

        if show_plot:
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            axs[0].plot(wieners_mean_batch_size)
            axs[0].plot(rho_mean_batch_size)
            axs[1].plot(abs_error)
            axs[1].plot(np.full(len(abs_error), abs_error_mean))
            trans = mtransforms.ScaledTranslation(-20/72,
                                                  7/72, fig.dpi_scale_trans)
            axs[1].text(0.0, 1.0, f'mean: {np.mean(abs_error)}%', transform=axs[1].transAxes + trans,
                        fontsize='medium', va='bottom', fontfamily='serif')

            plt.show()

        return abs_error, abs_error_mean


def base_gen(n=1000, dt = 0.001, v0 = 0.03, kappa = 0.07, 
        theta = 0.05, eta = 0.03, hurst = 0.5):


    new_params = {k: v for k, v in locals().items() if k != 'hurst'}
    new_params['hurst2'] = hurst
    hest = Heston(**new_params)
    cir = hest.model()[1]
    return cir


def gen(n = 1000, hurst = 0.5, lambd = 1, varkappa = 0.9, vartheta_vareta = 1):

    kappa = -np.log(varkappa) * n

    if vartheta_vareta < 0:
        vartheta = -vartheta_vareta
        vareta = 1
    else:
        vartheta = 1
        vareta = vartheta_vareta

    return base_gen(n = n, hurst = hurst, dt = 1, kappa = kappa, 
               theta = vartheta, eta = vareta)


#print(gen(vartheta_vareta = 0.5, n = 100))
#print(base_gen(kappa = 230))

    

