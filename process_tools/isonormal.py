from process_tools.abstract import Abstract
import numpy as np

class ISONormal(Abstract):
    def __init__(self, covariance_matrix):
        # legeneráljuk a gyökmátrix transzponáltat
        # csak egyszer kell
        self.covariance_matrix = covariance_matrix
        self.covariance_matrix_sqrt = self.get_covariance_matrix_sqrt()

    def get_covariance_matrix_sqrt(self):
        #start = time.time()
        #cholesky = np.linalg.cholesky(self.covariance_matrix)
        #print("cholesky", time.time() - start)
        return np.linalg.cholesky(self.covariance_matrix)

    def get_increments(self):
        return np.diff(self.covariance_matrix_sqrt @ np.random.normal(size=(self.covariance_matrix_sqrt.shape[0], 1)))

    def get_path(self):
        return self.covariance_matrix_sqrt @ np.random.normal(size=(self.covariance_matrix_sqrt.shape[0], 1))
