from process_generators.new_fbm_gen import FBM
import numpy as np
from abc import abstractmethod

class Abstract:
    @abstractmethod
    def get_covariance_matrix(self):
        pass

    @abstractmethod
    def get_covariance_matrix_sqrt(self):
        pass

    @abstractmethod
    def get_increments(self):
        pass

    @abstractmethod
    def get_path(self):
        pass

class FOU(Abstract):
    # Cholesky
    def __init__(self, hurst, alpha, sigma, initial_value=0, mu=0, time_grid=[], n=1000, T=1, isEquidistant=True):
        self.hurst = hurst
        self.n = n
        self.T = T
        self.isEquidistant = isEquidistant
        self.time_grid = self.time_grid_setter(time_grid)
        self.sigma = self.sigma_setter(sigma)
        self.alpha = self.alpha_setter(alpha)
        self.mu = mu
        self.initial_value = initial_value
        self.weights = self.integrand_setter()
        self.fbm = FBM(hurst=hurst, time_grid=time_grid,
                       T=T, n=n, isEquidistant=isEquidistant)
        self.delta_grid = self.delta_grid_setter()  # majd az Euler-sémához fog kelleni


    def delta_grid_setter(self):
        if self.isEquidistant:
            return np.diff(self.time_grid)

    def euler_scheme(self):
        fou_increments = np.zeros(self.n-1)
        path = np.zeros(self.n)
        path[0] = self.initial_value
        fbm_increments = self.fbm.get_increments()
        for i in range(self.n-1):
            fou_increments[i] = self.alpha * self.delta_grid[i] * \
                (self.mu - path[i]) + self.sigma * fbm_increments[i]
            path[i+1] = fou_increments[i] + path[i]
        return path

    def sigma_setter(self, sigma):
        if sigma > 0:
            return sigma
        else:
            raise Exception("The volatility parameter has to be positive.")

    def alpha_setter(self, alpha):
        if alpha > 0:
            return alpha
        else:
            raise Exception("The drift parameter has to be positive.")

    def time_grid_setter(self, time_grid):
        if self.isEquidistant:
            delta = self.T/(self.n-1)
            return [j*delta for j in range(0, self.n)]
        else:
            return time_grid

    def integrand_setter(self):
        # mu még nincs benne
        return [self.sigma*np.exp(-self.alpha*(self.T-self.time_grid[idx]))
            for idx in range(self.n-1)]

    def get_path(self):
        if self.initial_value == 0 and self.mu == 0:
            return np.cumsum([self.initial_value, *(self.weights * self.fbm.get_increments())])
        elif self.mu == 0:
            return [self.initial_value, *[np.exp(-self.alpha*self.time_grid[i]) * self.initial_value +
                                          (self.weights * self.fbm.get_increments())[i-1] for i in range(1, len(self.time_grid))]]
        else:
            return [self.initial_value, *[np.exp(-self.alpha*self.time_grid[i]) * self.initial_value + self.mu*(1-np.exp(-self.alpha*self.time_grid[i])) +
                                          (self.weights * self.fbm.get_increments())[i-1] for i in range(1, len(self.time_grid))]]

    def get_increment(self):
        return np.diff(self.get_path())


class FOU_noncond(Abstract):
    """
    Returns a realisation according to the "non-conditional", i.e. according to the joint distribution of the variables determined by the time_grid and the parameters
    """
    def __init__(self, hurst, alpha, sigma, initial_value=0, mu=0, time_grid=[], n=1000, T=1, isEquidistant=True):
        self.hurst = hurst
        self.n = n
        self.T = T
        self.isEquidistant = isEquidistant
        self.time_grid = self.time_grid_setter(time_grid)
        self.sigma = self.sigma_setter(sigma)
        self.alpha = self.alpha_setter(alpha)
        self.mu = mu
        self.initial_value = initial_value
        self.weights = self.integrand_setter()
        self.fbm = FBM(hurst=hurst, time_grid=time_grid,
                       T=T, n=n, isEquidistant=isEquidistant)
        self.delta_grid = self.delta_grid_setter()  # majd az Euler-sémához fog kelleni


    def delta_grid_setter(self):
        if self.isEquidistant:
            return np.diff(self.time_grid)

    def euler_scheme(self):
        fou_increments = np.zeros(self.n-1)
        path = np.zeros(self.n)
        path[0] = self.initial_value
        fbm_increments = self.fbm.get_increments()
        for i in range(self.n-1):
            fou_increments[i] = self.alpha * self.delta_grid[i] * \
                (self.mu - path[i]) + self.sigma * fbm_increments[i]
            path[i+1] = fou_increments[i] + path[i]
        return path

    def sigma_setter(self, sigma):
        if sigma > 0:
            return sigma
        else:
            raise Exception("The volatility parameter has to be positive.")

    def alpha_setter(self, alpha):
        if alpha > 0:
            return alpha
        else:
            raise Exception("The drift parameter has to be positive.")

    def time_grid_setter(self, time_grid):
        if self.isEquidistant:
            delta = self.T/(self.n-1)
            return [j*delta for j in range(0, self.n)]
        else:
            return time_grid

    def integrand_setter(self):
        # it has to be stored in a mtx in this case, since we aim at simulating according to the non-conditional joint distribution
        # this mtx will be multiplied by the increment vector
        disc_integrands = np.zeros((self.n-1, self.n-1))

        for i in range(self.n-1):
            for j in range(i+1):
                disc_integrands[i][j] = self.sigma*np.exp(-self.alpha*(self.time_grid[i]-self.time_grid[j]))


        return disc_integrands


    def get_path(self):
        if self.initial_value == 0 and self.mu == 0:
            return np.cumsum([self.initial_value, *(self.weights @ self.fbm.get_increments())])
        elif self.mu == 0:
            return [self.initial_value, *[np.exp(-self.alpha*self.time_grid[i]) * self.initial_value +
                                          (self.weights @ self.fbm.get_increments())[i-1] for i in range(1, len(self.time_grid))]]
        else:
            return [self.initial_value, *[np.exp(-self.alpha*self.time_grid[i]) * self.initial_value + self.mu*(1-np.exp(-self.alpha*self.time_grid[i])) +
                                          (self.weights @ self.fbm.get_increments())[i-1] for i in range(1, len(self.time_grid))]]

    def get_increment(self):
        return np.diff(self.get_path())

def gen(extra_batch_size = 1, n = 200, hurst = 0.7, varalpha = 0.5, lambd = 1, gamma_varsigma = 0.5, nu = 0.5):


    alpha = -np.log(varalpha) * (n - 1)

    if gamma_varsigma < 0:
        varsigma = 1
        gamma = 1 + gamma_varsigma
    else:
        gamma = 1
        varsigma = 1 - gamma_varsigma

    sigma = varsigma * (n - 1) ** hurst
    mu = nu
    initial_value = mu + gamma


    fou_gen = FOU_noncond(n = n, initial_value = initial_value, hurst = hurst,
                  alpha = alpha, sigma = sigma, T = 1)

    if extra_batch_size == 1:
        res = fou_gen.get_path()
    else:
        res = [fou_gen.get_path() for _ in range(extra_batch_size)]
    return res

