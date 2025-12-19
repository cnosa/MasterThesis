import numpy as np
from scipy import linalg


class LinearKalmanFilter:
    def __init__(self, theta, m0, P0, prior_logpdf,
                 A_fn, H_fn, Sigma_fn, Gamma_fn):
        self.theta = theta
        self.m = m0.copy()
        self.P = P0.copy()

        self.prior_logpdf = prior_logpdf
        self.A_fn = A_fn
        self.H_fn = H_fn
        self.Sigma_fn = Sigma_fn
        self.Gamma_fn = Gamma_fn

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

    def predict(self):
        A = np.atleast_2d(self.A_fn(self.theta))
        Sigma = np.atleast_2d(self.Sigma_fn(self.theta))

        self.m_minus = A @ self.m
        self.P_minus = A @ self.P @ A.T + Sigma

    def update(self, yk):
        H = np.atleast_2d(self.H_fn(self.theta))
        Gamma = np.atleast_2d(self.Gamma_fn(self.theta))

        mu_k = H @ self.m_minus
        S_k = H @ self.P_minus @ H.T + Gamma

        self.log_likelihood += self.log_gaussian_density(yk, mu_k, S_k)

        try:
            c, lower = linalg.cho_factor(S_k, check_finite=False)
            S_inv = linalg.cho_solve((c, lower), np.eye(S_k.shape[0]))
        except np.linalg.LinAlgError:
            S_inv = np.linalg.inv(S_k)

        K = self.P_minus @ H.T @ S_inv
        innovation = yk - mu_k

        self.m = self.m_minus + K @ innovation
        self.P = self.P_minus - K @ H @ self.P_minus

        # estabilidad numérica
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-8 * np.eye(self.P.shape[0])

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