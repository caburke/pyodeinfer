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
        self.tot_iter = self.burnin + self.thin*self.num_samples
        self.states_names = dsystem.initial_conditions.keys()
        self.num_states = len(self.state_names)
        self.par_dict = dsystem.pars
        self.num_pars = len(self.par_dict)
        