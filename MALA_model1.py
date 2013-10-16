#########################################################################
# Test MALA Program
#########################################################################

from __future__ import division
import copy as cp
import cPickle as pickle
import math as m
import matplotlib.pyplot as plt
import numpy as np
import PyDSTool as pd
import random as r
import scipy as sp
import scipy.stats

# Import from odeinfer module
import odeinfer
from odeinfer.simulate import *
from odeinfer.ode_models import *
from odeinfer.priordist import *
from odeinfer.likelihood import *

# Define Model
model1_ds = pd.Generator.Vode_ODEsystem(odeinfer.ode_models.model1_ds_args)
model1_traj = model1_ds.compute('model_1')
model1_sample = model1_traj.sample()

# Create Data
sim_times = np.arange(0, 201, 1)
obs_times = np.arange(32, 201, 1)
true_noise_scale = np.array([0.1, 0.1])
pure_obs, noisy_obs = sim_additive_normal_noise(model1_ds, obs_times, \
                                                true_noise_scale)

# Plot Data
#plt.figure(1)
#plt.subplot(211)
#plt.plot(model1_sample['t'], model1_sample['A'], c='b')
#plt.scatter(obs_times, noisy_obs[:, 0], c='b')
#plt.axis([0, 200, 0, 8])
#plt.title('Simulated and True Data')
#plt.subplot(212)
#plt.plot(model1_sample['t'], model1_sample['B'], c='B')
#plt.scatter(obs_times, noisy_obs[:, 1], c='B')
#plt.axis([0, 200, 0, 8])
#plt.show()

# Define MCMC Parameters
burnin = 0
thin = 1
num_samples= 20
num_iter = burnin + thin*num_samples
num_temp = 11    
num_states = len(model1_ds.initialconditions)
num_obs = len(obs_times)
state_name_list = model1_ds.variables.keys()
state_array = np.zeros((num_obs, num_states), dtype=float)
noisy_state_array = np.zeros((num_obs, num_states), dtype=float)

# Create Noiseless Observations
ds_traj = model1_ds.compute('simulate')
index = 0
for name in state_name_list:
    state_array[: , index ] = ds_traj(obs_times)[name]
    index += 1
num_parm = 14
cross_prob = 0.5
temp = np.array([pow(x/(num_temp - 1), 5) for x in range(num_temp - 1)])
temp = np.concatenate([temp, [1.]])

# Initial Proposal Distributions for parameters
# does not include cooperativity coefficients m and n
prop_dist0 = {'pars':{'nu': sp.stats.norm(0, 0.1),
                      'k0': sp.stats.norm(0, 0.1),
                      'k1': sp.stats.norm(0, 0.1),
                      'k2': sp.stats.norm(0, 0.1),
                      'k3': sp.stats.norm(0, 0.1),
                      'k4': sp.stats.norm(0, 0.1),
                      'Ka': sp.stats.norm(0, 0.1),
                      'Kb': sp.stats.norm(0, 0.1)},
             'init':{'A': sp.stats.norm(0, 0.02), 
                     'B': sp.stats.norm(0, 0.02)},
             'noise':{'A': sp.stats.norm(0, 0.001), 
                      'B': sp.stats.norm(0, 0.001)}}
prop_dist = {}
for i in range(num_temp):
    prop_dist[i] = prop_dist0

# Containers
par_history = np.zeros((num_samples, num_temp), 
                       dtype = {'names':['nu', 'k0', 'k1', 'k2', 'k3', 'k4', 'Ka', 'Kb'], 
                                'formats': np.repeat('float', 8)})
init_history = np.zeros((num_samples, num_temp), 
                        dtype = {'names':['A', 'B'], 
                                 'formats': np.repeat('float', 2)})
noise_history = np.zeros((num_samples, num_temp), 
                         dtype = {'names':['A', 'B'], 
                                  'formats': np.repeat('float', 2)})
sample_dict = {'pars': par_history,
               'init': init_history,
               'noise': noise_history}
# Dictionary with starting values for parameters at all temperatures
parm_dict = {'pars':{'nu': 1.177,
                      'k0': 0.0001,
                      'k1': 0.08,
                      'k2': 0.0482,
                      'k3': 1.605,
                      'k4': 0.535,
                      'Ka': 1.1,
                      'Kb': 3.0},
             'init':{'A': noisy_obs[0,0], 
                     'B': noisy_obs[0,1]},
             'noise':{'A': 0.1, 
                      'B': 0.1}}
cur_parm_dict = {}
for t in range(num_temp):
    cur_parm_dict[t] = {'pars':{'nu': 1.177,
                          'k0': 0.0001,
                          'k1': 0.08,
                          'k2': 0.0482,
                          'k3': 1.605,
                          'k4': 0.535,
                          'Ka': 1.1,
                          'Kb': 3.0},
                         'init':{'A': noisy_obs[0,0], 
                                 'B': noisy_obs[0,1]},
                         'noise':{'A': 0.1, 
                                  'B': 0.1}}

# Counters
accept_prop = {}
for t in range(num_temp):
    accept_prop[t] = {'pars':{'nu': 0,
                              'k0': 0,
                              'k1': 0,
                              'k2': 0,
                              'k3': 0,
                              'k4': 0,
                              'Ka': 0,
                              'Kb': 0},
                    'init':{'A': 0, 
                            'B': 0},
                    'noise':{'A': 0, 
                             'B': 0}}
attempt_prop = cp.deepcopy(accept_prop)
accept_ratio = {}
for t in range(num_temp):
    accept_ratio[t] = {'pars':{'nu': 0,
                              'k0': 0,
                              'k1': 0,
                              'k2': 0,
                              'k3': 0,
                              'k4': 0,
                              'Ka': 0,
                              'Kb': 0},
                    'init':{'A': 0, 
                            'B': 0},
                    'noise':{'A': 0, 
                             'B': 0}}
update_ctr = 0
pyds_error_ctr = 0

# Probabilistic Model
prior_dict = odeinfer.priordist.model1_ds_prior
log_likelihood = odeinfer.likelihood.normal_log_like