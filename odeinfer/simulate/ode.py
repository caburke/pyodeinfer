# Functions for simulating data from PyDSTool system

import numpy as np
import PyDSTool as pd

# Input should be PyDSTool dynamical system generator object
def sim_additive_normal_noise(dsystem, times, noise_scale, state_name_list):
    # Set up array to store simulated data
    num_states = len(dsystem.initialconditions)
    num_obs = len(times)
    state_array = np.zeros((num_obs, num_states), dtype=float)
    noisy_state_array = np.zeros((num_states, num_obs), dtype=float)
    
    # Create Noiseless Observations
    ds_traj = dsystem.compute('simulate')
    index = 0
    for name in state_name_list:
        state_array[: , index ] = ds_traj(times)[name]
        index += 1
    
    # Create Noise
    noise_mean = np.zeros(num_states, dtype=float)
    noise = np.random.multivariate_normal(noise_mean, np.diag(noise_scale), num_obs)
    noisy_state_array = state_array + noise
    
    return state_array, noisy_state_array