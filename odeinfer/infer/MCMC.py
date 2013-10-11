# Class and methods for MCMC Methods

class MCMC(dyn_system, prior, likelihood, \
            init_parm, init_ics, init_noise_sd, \
            burnin, num_samples, thin, kernel):
    def __init__(self):
        self.dyn_system = dyn_system
        self.prior = prior
        self.likelihood = likelihood
        self.init_parm = init_parm
        self.init_ics = init_ics
        self.init_noise_sd = init_noise_sd
        self.burnin = burnin
        self.num_samples = num_samples
        self.burnin = burnin
        self.thin = thin
        self.tot_iters = burnin + num_samples*thin
        self.kernel = kernel
    
    # Method runs the MCMC algorithm
    def run(self):
        
               
