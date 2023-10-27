import numpy as np
from isonormal import ISONormal
from process_tools.ltm import LTMatrix
# a szimulálás [0,1]-en történik és skálázással kapjuk vissza a [0,T]-re


class FBM(ISONormal):
    def __init__(self, hurst, time_grid=[], n=1000, T=1, isEquidistant=True, *args, **kwargs) -> None:
        self.hurst = hurst
        self.time_grid = time_grid
        self.isEquidistant = isEquidistant
        self.n = n
        self.T = T
        self.timegrid = self.time_grid_setter(time_grid)
        self.acf_vec = self.acf_setter()  # ekvi esetben gyorsabb metódus
        self.covariance_matrix = self.cov_mtx_setter()
        super().__init__(self.covariance_matrix)

    def acf_setter(self):
        return [1] + [0.5*((k-1)**(2*self.hurst)+(k+1)**(2*self.hurst)
                    - 2*(k)**(2*self.hurst)) for k in range(1, self.n)]

    def cov_mtx_setter(self):
        ltm = LTMatrix(self.acf_vec).get()
        return ltm

    def time_grid_setter(self, time_grid):
        if self.isEquidistant:
            delta = self.T/(self.n-1)

            # return np.linspace(0, self.T, self.n)
            return [j*delta for j in range(0, self.n)]
        else:
            return time_grid

    def get_increments_default(self):
        # skálázás [0,1]-re, ekvin szimulál
        std_normal_vec = np.random.normal(size=self.n)
        return self.covariance_matrix_sqrt @ std_normal_vec

    def get_increments(self):
        # [0,T]-n általánosan
        return self.get_path(increments=True)

    def get_path(self, pair=False, increments=False):
        # skálázás [0,T]-re, nem ekvin vigyázni
        # feltesszük, hogy nullából indul
        # pair - Boolean -> path-increment párt vagy szimplán a path-t

        # külön kell kezelni az ekvit és nem ekvit


        incs = self.get_increments_default()
        path = np.zeros(self.n)
        for i in range(1, self.n):
            path[i] = incs[i-1] + path[i-1]
        

        # [0,1] path -> [0,T] path
        if self.isEquidistant and not self.T == 1:
            path = path*(self.T)**self.hurst
        elif not self.isEquidistant:
            for j in range(1,self.n):
                #skálázás
                path[j] = ((self.n-1) * self.time_grid[j]/(j))**self.hurst * path[j]


        # visszadiff az increments-hez
        if not pair and not increments:
            return path
        elif pair:
            return (np.diff(path), path)
        elif increments:
            return np.diff(path)
