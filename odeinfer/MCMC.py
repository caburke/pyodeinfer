# Classes for MCMC algorithms

def class MCMC(dsystem, prior, likelihood,
               num_samples, burnin, thin):
    def _init_(self):
        self.dsystem = dsystem
        self.prior = prior
        self.likelihood = likelihood
        self.num_samples = num_samples
        self.burnin = burnin
        self.thin = thin
    