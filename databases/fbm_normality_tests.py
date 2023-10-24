import numpy as np
from scipy import stats
import matplotlib.pylab as plt


def normality_fbm(path, method="shapiro-wilks"):
    diff_path = np.diff(path)

    if method == "shapiro-wilks":

        stat, p = stats.shapiro(diff_path)

        return {"statistic": stat, "p_value": p}

    elif method == 'normal':
        """
        This function tests the null hypothesis that a sample comes from a normal distribution. 
        It is based on D’Agostino and Pearson’s [1], [2] test that combines skew and kurtosis
        to produce an omnibus test of normality.

        [1] D’Agostino, R. B. (1971), “An omnibus test of normality for moderate and large sample size”, Biometrika, 58, 341-348
        [2] D’Agostino, R. and Pearson, E. S. (1973), “Tests for departure from normality”, Biometrika, 60, 613-622
        """
        stat, p = stats.normaltest(diff_path)

        return {"statistic": stat, "p_value": p}

    elif method == 'histogram':
        """
        ImportError: cannot import name ‘_np_version_under1p14’ from ‘pandas.compat.numpy’ (unknown location)
        """
        # sns.distplot(self._diff_path)
        plt.hist(diff_path)
        plt.title('Histogram')
        plt.show()

    elif method == 'qqplot':
        stats.probplot(diff_path, dist="norm", plot=plt)
        plt.show()

    else:
        pass
