import numpy as np


class Covariance2D:
    """
    fonction de covariance 2D (temps/espace)
    """

    def __init__(
        self,
        variance: float,
        lambda_space: float,
        threshold: float = 1e-3,
    ) -> None:
        """ constructor """
        self._variance = variance
        self._lambda_s = lambda_space
        self._threshold = threshold

    def _space_cov(self, delta: np.ndarray) -> np.ndarray:
        """ space covariance function """
        return np.exp(-0.5 * (np.abs(delta) / self.lambda_s) ** 2)

    def _time_cov(self, delta: np.ndarray) -> np.ndarray:
        """ time covariance function """
        raise NotImplementedError()

    def cov(self, lon: np.ndarray, lat: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        calcul de la matrice de covariance d'après les positions des observastions
        """

        xx, yy, tt = np.meshgrid(lon, lat, time)
        ds = self._lonlat_to_distance(xx, yy)
        dt = self._time_to_distance(tt)

        return self.var * self._space_cov(ds) * self._time_cov(dt)

    def _lonlat_to_distance(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        """
        conversion lon/lat vers une matrice de distances
        """
        xM, yM = np.meshgrid(xx.flatten(), yy.flatten(), indexing="ij")
        dxM = xM - xM.T
        dyM = yM - yM.T
        # distance dans un espace cartésien (en degrés)
        # TODO: passer à une estimation des distances en km
        return np.sqrt(dxM ** 2 + dyM ** 2)

    def _time_to_distance(self, tt: np.ndarray) -> np.ndarray:
        """
        conversion temps vers une matrice de distance
        """
        tM, _ = np.meshgrid(tt.flatten(), tt.flatten(), indexing="ij")
        return np.subtract(tM, np.transpose(tM))

    @property
    def threshold(self):
        return self._threshold

    @property
    def lambda_t(self):
        return self._lambda_t

    @property
    def lambda_s(self):
        return self._lambda_s

    @property
    def var(self):
        return self._variance


class NoiseCovariance2D(Covariance2D):
    """
    2D noise covariance
    """

    def __init__(
        self,
        variance: float,
        lambda_space: float,
        lambda_time: float,
        threshold: float = 0.001,
    ) -> None:
        super().__init__(variance, lambda_space, threshold)
        self._lambda_t = lambda_time

    def _time_cov(self, delta: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (np.abs(delta) / self.lambda_t) ** 2)