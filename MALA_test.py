#########################################################################
# Test Metropolis Adjusted MCMC Program
#########################################################################

# Import from Python 3.x
from __future__ import division

# Import other modules
import numpy as np
import PyDSTool as pd

# Import my modules
from odeinfer.models import fitzhugh_nagumo
from odeinfer.simulate import ode

# We are using the Fitzhugh-Nagumo ODEs to model the data
# This generator uses the VODE solver
fhn_ds = pd.Generator.Vode_ODEsystem(fitzhugh_nagumo.fhn_ds_args)

# Simulate the data from the FN ODEs at obs_times
obs_times = np.linspace(0, 20, 21)
noise_scale = [0.1, 0.1]
state_names = ['V', 'R']  # This is the order of the states in the returned  array
pure_obs, noisy_obs = ode.sim_additive_normal_noise(fhn_ds, obs_times, noise_scale, state_names)