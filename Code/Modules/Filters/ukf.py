import numpy as np
from scipy import linalg


class UnscentedKalmanFilter:
    def __init__(self, theta, m0, P0, prior_logpdf,
                 Phi, h, Sigma_fn, Gamma_fn,
                 alpha=1e-3, beta=2.0, kappa=0.0):

        self.theta = theta
        self.m = m0.astype(np.float64).copy()
        self.P = P0.astype(np.float64).copy()
        self.d = self.m.shape[0]

        self.prior_logpdf = prior_logpdf
        self.Phi = Phi
        self.h = h
        self.Sigma_fn = Sigma_fn
        self.Gamma_fn = Gamma_fn

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambd = alpha**2 * (self.d + kappa) - self.d
        self.denom = self.d + self.lambd
        self.sqrt_d_lamb = np.sqrt(self.denom)

        self.w0m = self.lambd / self.denom
        self.w0c = self.w0m + (1.0 - alpha**2 + beta)
        self.wim = 1.0 / (2.0 * self.denom)

        self.log_prior = prior_logpdf(theta)
        self.log_likelihood = 0.0


    @staticmethod
    def log_gaussian_density(y, mu, S):
        d = len(y)
        diff = y - mu
        try:
            c, lower = linalg.cho_factor(S, check_finite=False)
            alpha = linalg.cho_solve((c, lower), diff)
            logdet = 2 * np.sum(np.log(np.diag(c)))
        except np.linalg.LinAlgError:
            alpha = np.linalg.solve(S, diff)
            logdet = np.log(np.linalg.det(S))

        return -0.5 * (diff.T @ alpha + logdet + d * np.log(2 * np.pi))
    
    def _sigma_points(self, m, P):
        try:
            A = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            A = np.linalg.cholesky(P + 1e-8 * np.eye(self.d))

        X = np.empty((self.d, 2 * self.d + 1))
        X[:, 0] = m

        for i in range(self.d):
            col = A[:, i]
            X[:, 1 + i] = m + self.sqrt_d_lamb * col
            X[:, 1 + self.d + i] = m - self.sqrt_d_lamb * col

        return X
    
    def predict(self):
        X = self._sigma_points(self.m, self.P)

        X_hat = np.empty_like(X)
        for i in range(2 * self.d + 1):
            X_hat[:, i] = self.Phi(X[:, i], self.theta)

        m_minus = self.w0m * X_hat[:, 0] + self.wim * np.sum(X_hat[:, 1:], axis=1)

        P_minus = np.atleast_2d(self.Sigma_fn(self.theta)).astype(np.float64)
        for i in range(2 * self.d + 1):
            diff = X_hat[:, i] - m_minus
            P_minus += (self.w0c if i == 0 else self.wim) * np.outer(diff, diff)

        self.m_minus = m_minus
        self.P_minus = 0.5 * (P_minus + P_minus.T)

    def update(self, yk):
        m = yk.shape[0]

        X = self._sigma_points(self.m_minus, self.P_minus)

        Y_hat = np.empty((m, 2 * self.d + 1))
        for i in range(2 * self.d + 1):
            Y_hat[:, i] = self.h(X[:, i], self.theta)

        mu_k = self.w0m * Y_hat[:, 0] + self.wim * np.sum(Y_hat[:, 1:], axis=1)

        S_k = np.atleast_2d(self.Gamma_fn(self.theta)).astype(np.float64)
        for i in range(2 * self.d + 1):
            dy = Y_hat[:, i] - mu_k
            S_k += (self.w0c if i == 0 else self.wim) * np.outer(dy, dy)

        S_k = 0.5 * (S_k + S_k.T)

        self.log_likelihood += self.log_gaussian_density(yk, mu_k, S_k)

        C_k = np.zeros((self.d, m))
        for i in range(2 * self.d + 1):
            dx = X[:, i] - self.m_minus
            dy = Y_hat[:, i] - mu_k
            C_k += (self.w0c if i == 0 else self.wim) * np.outer(dx, dy)

        try:
            cho = np.linalg.cholesky(S_k)
            K = linalg.cho_solve((cho, True), C_k.T).T
        except np.linalg.LinAlgError:
            K = C_k @ np.linalg.inv(S_k)

        self.m = self.m_minus + K @ (yk - mu_k)
        self.P = self.P_minus - K @ C_k.T
        self.P = 0.5 * (self.P + self.P.T)

    def filter(self, Y):
        Y = np.asarray(Y)

        for yk in Y:
            self.predict()
            self.update(yk)

        return {
            "log_prior": self.log_prior,
            "log_likelihood": self.log_likelihood,
            "log_posterior": self.log_prior + self.log_likelihood,
            "prediction": (self.m, self.P)
        }