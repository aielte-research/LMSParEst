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
