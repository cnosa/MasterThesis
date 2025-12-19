import numpy as np
from scipy.special import logsumexp

class MonteCarloMHFilter:
    def __init__(self, theta,
                 N_pred, N_mcmc,
                 prior_sampler,
                 transition_sampler,
                 log_likelihood,
                 proposal_sampler,
                 proposal_logpdf,
                 burn_in=500,
                 thinning=5,
                 ess_threshold=0.5):

        self.theta = theta

        # tamaños
        self.N_pred = N_pred
        self.N_mcmc = N_mcmc

        # modelo
        self.prior_sampler = prior_sampler
        self.transition_sampler = transition_sampler
        self.log_likelihood = log_likelihood

        # MH
        self.proposal_sampler = proposal_sampler
        self.proposal_logpdf = proposal_logpdf
        self.burn_in = burn_in
        self.thinning = thinning
        self.ess_threshold = ess_threshold

        # estado interno
        self.particles = None
        self.log_likelihood_total = 0.0


    def initialize(self, N_init):
        self.particles = np.array([
            self.prior_sampler() for _ in range(N_init)
        ])

    def predict(self):
        idx = np.random.choice(
            len(self.particles),
            size=self.N_pred,
            replace=True
        )

        x_prev = self.particles[idx]

        self.predicted_particles = np.array([
            self.transition_sampler(x, self.theta)
            for x in x_prev
        ])

    def log_posterior_state(self, x, y):
        # prior empírico uniforme
        log_prior = -np.log(len(self.predicted_particles))
        return self.log_likelihood(y, x, self.theta) + log_prior

    def update(self, y):
        samples = []
        logps = []

        # inicialización MH
        x_current = self.predicted_particles[
            np.random.randint(len(self.predicted_particles))
        ]
        logp_current = self.log_posterior_state(x_current, y)

        total_iters = self.burn_in + self.N_mcmc * self.thinning

        for it in range(total_iters):
            x_prop = self.proposal_sampler(x_current)
            logp_prop = self.log_posterior_state(x_prop, y)

            log_alpha = (
                logp_prop
                + self.proposal_logpdf(x_current, x_prop)
                - logp_current
                - self.proposal_logpdf(x_prop, x_current)
            )

            if np.log(np.random.rand()) < log_alpha:
                x_current = x_prop
                logp_current = logp_prop

            if it >= self.burn_in and (it - self.burn_in) % self.thinning == 0:
                samples.append(x_current)
                logps.append(logp_current)

        samples = np.array(samples)
        logps = np.array(logps)

        logw = logps - logsumexp(logps)
        w = np.exp(logw)

        ess = 1.0 / np.sum(w**2)
        ess_rel = ess / len(w)

        if ess_rel < self.ess_threshold:
            raise RuntimeError(
                f"ESS relativo insuficiente: {ess_rel:.2f}"
            )

        self.particles = samples
        self.log_likelihood_total += logsumexp(logps) - np.log(len(logps))

    def step(self, y):
        self.predict()
        self.update(y)

    def filter(self, Y):
        for y in Y:
            self.step(y)

        mean = np.mean(self.particles, axis=0)
        diff = self.particles - mean
        cov = diff.T @ diff / len(self.particles)

        return {
            "log_likelihood": self.log_likelihood_total,
            "prediction": (mean, cov),
            "particles": self.particles
        }
