import numpy as np
from scipy import linalg

class LinearStateSpaceModel:
    def __init__(self, A, H, Sigma, Gamma):
        self.A = np.atleast_2d(A)
        self.H = np.atleast_2d(H)
        self.Sigma = np.atleast_2d(Sigma)
        self.Gamma = np.atleast_2d(Gamma)

        self.d = self.A.shape[0]
        self.m = self.H.shape[0]

    def transition(self, x):
        return self.A @ x

    def observation(self, x):
        return self.H @ x

    def sample_process_noise(self):
        return np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=self.Sigma
        )

    def sample_observation_noise(self):
        return np.random.multivariate_normal(
            mean=np.zeros(self.m),
            cov=self.Gamma
        )

    def sample_transition(self, x):
        return self.transition(x) + self.sample_process_noise()

    def sample_observation(self, x):
        return self.observation(x) + self.sample_observation_noise()

    @staticmethod
    def log_gaussian(y, mu, S):
        d = len(y)
        diff = y - mu
        try:
            c, lower = linalg.cho_factor(S, check_finite=False)
            alpha = linalg.cho_solve((c, lower), diff)
            logdet = 2 * np.sum(np.log(np.diag(c)))
        except np.linalg.LinAlgError:
            alpha = np.linalg.solve(S, diff)
            logdet = np.log(np.linalg.det(S))

        return -0.5 * (diff @ alpha + logdet + d * np.log(2 * np.pi))

    def log_transition_density(self, x_next, x):
        mu = self.transition(x)
        return self.log_gaussian(x_next, mu, self.Sigma)

    def log_observation_density(self, y, x):
        mu = self.observation(x)
        return self.log_gaussian(y, mu, self.Gamma)

    def simulate(self, T, x0):
        X = np.zeros((T + 1, self.d))
        Y = np.zeros((T, self.m))

        X[0] = x0

        for t in range(T):
            X[t + 1] = self.sample_transition(X[t])
            Y[t] = self.sample_observation(X[t + 1])

        return X, Y

    def predict_distribution(self, m, P):
        m_pred = self.A @ m
        P_pred = self.A @ P @ self.A.T + self.Sigma
        return m_pred, P_pred

    def exact_loglikelihood(self, Y, m0, P0):
        m, P = m0.copy(), P0.copy()
        logL = 0.0

        for y in Y:
            m_pred, P_pred = self.predict_distribution(m, P)

            S = self.H @ P_pred @ self.H.T + self.Gamma
            mu = self.H @ m_pred

            logL += self.log_gaussian(y, mu, S)

            K = P_pred @ self.H.T @ np.linalg.inv(S)
            m = m_pred + K @ (y - mu)
            P = P_pred - K @ self.H @ P_pred

        return logL
