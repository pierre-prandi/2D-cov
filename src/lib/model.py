import numpy as np


class Model:
    def __init__(self, order: int) -> None:
        self.order = order

    def elementary_design_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        create an elementary design matrix
        """
        X = np.empty((len(x), self.order + 1))
        for i in np.arange(self.order + 1):
            X[:, i] = x ** i
        return X

    def design(
        self, lon: np.ndarray, lat: np.ndarray, time: np.ndarray, values: np.ndarray
    ) -> None:
        """
        create system design matrix
        """
        xx, yy, tt = np.meshgrid(lon, lat, time, indexing='ij')
        ntimes = len(time)
        nobs = np.prod(np.shape(xx))
        print("model::design::nobs: ", nobs)
        print("model::design::xx.shape: ", xx.shape)
        
        npos = len(lon) * len(lat)
        nparams = npos * (self.order + 1)

        X = np.ones((nobs, self.order + 1))
        Y = np.zeros((nobs))

        idx = 0
        for i in range(xx.shape[0]):
            print(f'i:{i}')
            for j in range(xx.shape[1]):
                print(f'j:{j}')
                edm = self.elementary_design_matrix(tt[i, j, :])
                X[idx : idx + ntimes, :] = edm
                Y[idx : idx + ntimes] = values[i, j, :]
                idx += ntimes

        self.X = X
        self.Y = Y

    def fit(self):
        """
        perform the inversion
        """
        XtX = np.dot(np.transpose(self.X), self.X)
        XtX_inv = np.linalg.inv(XtX)
        self.beta_hat = np.dot(XtX_inv, np.dot(np.transpose(self.X), self.Y))

    def parameters(self) -> np.ndarray:
        """
        return model parameters
        """
        return self.beta_hat

    def parameters_variance(self, omega: np.ndarray) -> np.ndarray:
        """
        return variance of model parameters
        """
        XtX = np.dot(np.transpose(self.X), self.X)
        XtX_inv = np.linalg.inv(XtX)
        var_beta_hat = np.diag(
            np.dot(np.dot(np.dot(np.dot(XtX_inv, self.X.T), omega), self.X), XtX_inv)
        )
        return var_beta_hat

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        perform model prediction
        """
        X = self.elementary_design_matrix(x)
        return np.dot(X, self.beta_hat)