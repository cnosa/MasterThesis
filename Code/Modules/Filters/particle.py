import numpy as np
from scipy.special import logsumexp

class ParticleFilter:
    def __init__(self, theta, N, m0, P0, prior_logpdf,
                 Phi, h, Sigma_fn, Gamma_fn,
                 resample_threshold=0.5):

        self.theta = theta
        self.N = N
        self.prior_logpdf = prior_logpdf

        self.Phi = Phi
        self.h = h
        self.Sigma = np.atleast_2d(Sigma_fn(theta))
        self.Gamma = np.atleast_2d(Gamma_fn(theta))

        self.d = m0.shape[0]
        self.m = m0.copy()
        self.P = P0.copy()

        # Inicialization of particles
        self.particles = np.random.multivariate_normal(
            mean=m0, cov=P0, size=N
        )

        self.logw = np.zeros(N) - np.log(N)

        self.log_prior = prior_logpdf(theta)
        self.log_likelihood = 0.0

        self.resample_threshold = resample_threshold

    @staticmethod
    def log_gaussian_density(y, mu, S):
        d = len(y)
        diff = y - mu
        try:
            c = np.linalg.cholesky(S)
            alpha = np.linalg.solve(c.T, np.linalg.solve(c, diff))
            logdet = 2 * np.sum(np.log(np.diag(c)))
        except np.linalg.LinAlgError:
            alpha = np.linalg.solve(S, diff)
            logdet = np.log(np.linalg.det(S))

        return -0.5 * (diff @ alpha + logdet + d * np.log(2 * np.pi))

    def predict(self):
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.d), cov=self.Sigma, size=self.N
        )

        for i in range(self.N):
            self.particles[i] = self.Phi(self.particles[i], self.theta) + noise[i]

    def update(self, yk):
        logw_new = np.zeros(self.N)

        for i in range(self.N):
            y_pred = self.h(self.particles[i], self.theta)
            logw_new[i] = self.logw[i] + self.log_gaussian_density(
                yk, y_pred, self.Gamma
            )

        # stable normalization
        logZ = logsumexp(logw_new)
        self.logw = logw_new - logZ
        self.log_likelihood += logZ

    def effective_sample_size(self):
        w = np.exp(self.logw)
        return 1.0 / np.sum(w**2)

    def resample(self):
        w = np.exp(self.logw)
        idx = np.random.choice(self.N, size=self.N, p=w)

        self.particles = self.particles[idx]
        self.logw[:] = -np.log(self.N)

    def estimate(self):
        w = np.exp(self.logw)
        mean = np.sum(self.particles * w[:, None], axis=0)

        diff = self.particles - mean
        cov = diff.T @ (diff * w[:, None])

        self.m = mean
        self.P = 0.5 * (cov + cov.T)

    def step(self, yk):
        self.predict()
        self.update(yk)

        if self.effective_sample_size() < self.resample_threshold * self.N:
            self.resample()

        self.estimate()

    def filter(self, Y):
        Y = np.asarray(Y)

        for yk in Y:
            self.step(yk)

        return {
            "log_prior": self.log_prior,
            "log_likelihood": self.log_likelihood,
            "log_posterior": self.log_prior + self.log_likelihood,
            "prediction": (self.m, self.P),
            "particles": self.particles,
            "weights": np.exp(self.logw)
        }
