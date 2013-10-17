#########################################################################
# Test Population MCMC Program
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
model1_ds = pd.Generator.Dopri_ODEsystem(odeinfer.ode_models.model1_ds_args)
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
burnin = 100000
thin = 10
num_samples= 10000
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
    accept_ratio[t] = {'pars':{'nu': [],
                              'k0': [],
                              'k1': [],
                              'k2': [],
                              'k3': [],
                              'k4': [],
                              'Ka': [],
                              'Kb': []},
                    'init':{'A': [], 
                            'B': []},
                    'noise':{'A': [], 
                             'B': []}}
update_ctr = 0
pyds_error_ctr = 0

# Probabilistic Model
prior_dict = odeinfer.priordist.model1_ds_prior
log_likelihood = odeinfer.likelihood.normal_log_like
    
#########################################################################
# Population MCMC Code
#########################################################################

# Values for Iteration 0

cur_lpval = np.zeros(num_temp)
cur_llval = np.zeros(num_temp)
cur_lpstval = np.zeros(num_temp)
cur_obs = np.zeros((num_temp, pure_obs.shape[0], pure_obs.shape[1]))
for t in range(num_temp):
    cur_obs[t] = pure_obs

for t in range(num_temp):
    cur_lpval[t] = log_prior_pdf(prior_dict, cur_parm_dict[t])
    cur_llval[t] = log_likelihood(noisy_obs, pure_obs, cur_parm_dict[t])
    cur_lpstval[t] = cur_lpval[t] + temp[t]*cur_llval[t]
    
# Start Loop!
for i in xrange(num_samples):
    
    if i % 10 == 0:
        print 'Iteration', i
    
    # Mutation Step
    rand_temp = np.random.randint(0, num_temp)
    # Update Model Parameters
    for p in parm_dict['pars'].iterkeys():
        # Mutate ODE parameters and Update Prior
        prop_parm = cp.deepcopy(cur_parm_dict[rand_temp])
        prop_parm['pars'][p] += np.asscalar(prop_dist[rand_temp]['pars'][p].rvs(1))
        prop_lpval = log_prior_pdf(prior_dict, prop_parm)
        
        # Update DS
        model1_ds.set(pars = prop_parm['pars'], ics = prop_parm['init'])
        try:
            prop_traj = model1_ds.compute('model1')
            prop_obs = np.transpose(np.array([prop_traj(obs_times)['A'], 
                                              prop_traj(obs_times)['B']]))
            # Update Likelihood and Posterior
            prop_llval = log_likelihood(noisy_obs, prop_obs, prop_parm)
            prop_lpstval = prop_lpval + temp[rand_temp]*prop_llval

            # Accept/Reject Mutation
            if prop_lpstval - cur_lpstval[rand_temp] > m.log(r.random()):
                cur_parm_dict[rand_temp]['pars'][p] = cp.deepcopy(prop_parm['pars'][p])
                cur_lpval[rand_temp] = prop_lpval
                cur_llval[rand_temp] = prop_llval
                cur_obs[rand_temp] = prop_obs
                cur_lpstval[rand_temp] = prop_lpstval
                accept_prop[rand_temp]['pars'][p] += 1
            attempt_prop[rand_temp]['pars'][p] += 1
     
        except (pd.PyDSTool_ExistError, TypeError):
            pyds_error_ctr += 1
        
   
    # Update Initial Conditions
    for init in parm_dict['init'].iterkeys():
        # Mutate ODE initial conditions and Update Prior
        prop_parm = cp.deepcopy(cur_parm_dict[rand_temp])
        prop_parm['init'][init] += np.asscalar(prop_dist[rand_temp]['init'][init].rvs(1))
        prop_lpval = log_prior_pdf(prior_dict, prop_parm)
        
        # Update DS
        model1_ds.set(pars = prop_parm['pars'], ics = prop_parm['init'])
        try:        
            prop_traj = model1_ds.compute('model1')
            prop_obs = np.transpose(np.array([prop_traj(obs_times)['A'], 
                                              prop_traj(obs_times)['B']]))
            
            # Update Likelihood and Posterior
            prop_llval = log_likelihood(noisy_obs, prop_obs, prop_parm)
            prop_lpstval = prop_lpval + temp[rand_temp]*prop_llval
            
            # Accept/Reject Mutation
            if prop_lpstval - cur_lpstval[rand_temp] > m.log(r.random()):
                cur_parm_dict[rand_temp]['init'][init] = cp.deepcopy(prop_parm['init'][init])
                cur_lpval[rand_temp] = prop_lpval
                cur_llval[rand_temp] = prop_llval
                cur_obs[rand_temp] = prop_obs
                cur_lpstval[rand_temp] = prop_lpstval
                accept_prop[rand_temp]['init'][init] += 1
            attempt_prop[rand_temp]['init'][init] += 1
        except (pd.PyDSTool_ExistError, TypeError):
            pyds_error_ctr += 1
        
    # Update Noise Parameters
    for n in parm_dict['noise'].iterkeys():
        # Mutate noise scale parameters and Update Prior
        prop_parm = cp.deepcopy(cur_parm_dict[rand_temp])
        prop_parm['noise'][n] += np.asscalar(prop_dist[rand_temp]['noise'][n].rvs(1))
        prop_lpval = log_prior_pdf(prior_dict, prop_parm)
        
        # No need to update DS for noise parameter changes
        
        # Update Likelihood and Posterior
        prop_obs = cp.deepcopy(cur_obs[rand_temp])
        prop_llval = log_likelihood(noisy_obs, prop_obs, prop_parm)
        prop_lpstval = prop_lpval + temp[rand_temp]*prop_llval
        
        # Accept/Reject Mutation
        if prop_lpstval - cur_lpstval[rand_temp] > m.log(r.random()):
            cur_parm_dict[rand_temp]['noise'][n] = cp.deepcopy(prop_parm['noise'][n])
            cur_lpval[rand_temp] = prop_lpval
            cur_llval[rand_temp] = prop_llval
            cur_obs[rand_temp] = prop_obs
            cur_lpstval[rand_temp] = prop_lpstval
            accept_prop[rand_temp]['noise'][n] += 1
        attempt_prop[rand_temp]['noise'][n] += 1
        
      
    rand_num = r.random()  
    # Crossover
    if rand_num < cross_prob:
        rand_cross = r.sample(range(num_temp), 2)
        # Temperatures where a crossover is proposed
        rand_temp1 = rand_cross[0]
        rand_temp2 = rand_cross[1]
        rand_parm_int = r.choice(range(1, len(parm_dict['pars']) + 1))
        rand_init_int = r.choice(range(1, len(parm_dict['init']) + 1))
        rand_noise_int = r.choice(range(1, len(parm_dict['noise']) + 1))
        rand_parm = r.sample(parm_dict['pars'].keys(), rand_parm_int)
        rand_init = r.sample(parm_dict['init'].keys(), rand_init_int)
        rand_noise = r.sample(parm_dict['noise'].keys(), rand_noise_int)
        prop_cross1 = cp.deepcopy(cur_parm_dict[rand_temp1])
        prop_cross2 = cp.deepcopy(cur_parm_dict[rand_temp2])
        for rp in rand_parm:
            prop_cross1['pars'][rp] = cur_parm_dict[rand_temp2]['pars'][rp]
            prop_cross2['pars'][rp] = cur_parm_dict[rand_temp1]['pars'][rp]
        for ri in rand_init:
            prop_cross1['init'][ri] = cur_parm_dict[rand_temp2]['init'][ri]
            prop_cross2['init'][ri] = cur_parm_dict[rand_temp1]['init'][ri]
        for rn in rand_noise:
            prop_cross1['noise'][rn] = cur_parm_dict[rand_temp2]['noise'][rn]
            prop_cross2['noise'][rn] = cur_parm_dict[rand_temp1]['noise'][rn]
        
        
        try:
            # Calculate quantities to determine crossover acceptance
            prop_lpval1 = log_prior_pdf(prior_dict, prop_cross1)
            model1_ds.set(pars = prop_cross1['pars'], ics = prop_cross1['init'])
            prop_traj1 = model1_ds.compute('model1')
            prop_obs1 = np.transpose(np.array([prop_traj1(obs_times)['A'], 
                                               prop_traj1(obs_times)['B']]))
            prop_llval1 = log_likelihood(noisy_obs, prop_obs1, prop_cross1)
            prop_lpstval1 = prop_lpval1 + temp[rand_temp1]*prop_llval1
            
            prop_lpval2 = log_prior_pdf(prior_dict, prop_cross2)
            model1_ds.set(pars = prop_cross2['pars'], ics = prop_cross2['init'])
            prop_traj2 = model1_ds.compute('model1')
            prop_obs2 = np.transpose(np.array([prop_traj2(obs_times)['A'], 
                                               prop_traj2(obs_times)['B']]))
            prop_llval2 = log_likelihood(noisy_obs, prop_obs2, prop_cross2)
            prop_lpstval2 = prop_lpval2 + temp[rand_temp1]*prop_llval2
            
            # Determine Accepteance of Crossover Move
            if (prop_lpstval1 + prop_lpstval2) - (cur_lpstval[rand_temp1] + cur_lpstval[rand_temp1]) > m.log(r.random()):
                cur_parm_dict[rand_temp1] = cp.deepcopy(prop_cross1)
                cur_lpval[rand_temp1] = prop_lpval1
                cur_obs[rand_temp1] = prop_obs1
                cur_llval[rand_temp1] = prop_llval1
                cur_lpstval[rand_temp1] = prop_lpstval1
                
                cur_parm_dict[rand_temp2] = cp.deepcopy(prop_cross2)
                cur_lpval[rand_temp2] = prop_lpval2
                cur_obs[rand_temp2] = prop_obs2
                cur_llval[rand_temp2] = prop_llval2
                cur_lpstval[rand_temp2] = prop_lpstval2
        except (pd.PyDSTool_ExistError, TypeError):
            pyds_error_ctr += 1
    
    # Exchange
    else:
        rand_temp1 = r.choice(range(0, num_temp - 1))
        rand_temp2 = rand_temp1 + 1
        prop_exc1 = cp.deepcopy(cur_parm_dict[rand_temp2])
        prop_exc2 = cp.deepcopy(cur_parm_dict[rand_temp1])
        
        try:
            # Calculate quantities to determine exchange acceptance
            prop_lpval1 = log_prior_pdf(prior_dict, prop_exc1)
            model1_ds.set(pars = prop_exc1['pars'], ics = prop_exc1['init'])
            prop_traj1 = model1_ds.compute('model1')
            prop_obs1 = np.transpose(np.array([prop_traj1(obs_times)['A'], 
                                               prop_traj1(obs_times)['B']]))
            prop_llval1 = log_likelihood(noisy_obs, prop_obs1, prop_exc1)
            prop_lpstval1 = prop_lpval1 + temp[rand_temp1]*prop_llval1
            
            prop_lpval2 = log_prior_pdf(prior_dict, prop_exc2)
            model1_ds.set(pars = prop_exc2['pars'], ics = prop_exc2['init'])
            prop_traj2 = model1_ds.compute('model1')
            prop_obs2 = np.transpose(np.array([prop_traj2(obs_times)['A'], 
                                               prop_traj2(obs_times)['B']]))
            prop_llval2 = log_likelihood(noisy_obs, prop_obs2, prop_exc2)
            prop_lpstval2 = prop_lpval2 + temp[rand_temp1]*prop_llval2
        
            # Determine Acceptance of Exchange Move
            if (prop_lpstval1 + prop_lpstval2) - (cur_lpstval[rand_temp1] + cur_lpstval[rand_temp1]) > m.log(r.random()):
                cur_parm_dict[rand_temp1] = cp.deepcopy(prop_exc1)
                cur_lpval[rand_temp1] = prop_lpval1
                cur_obs[rand_temp1] = prop_obs1
                cur_llval[rand_temp1] = prop_llval1
                cur_lpstval[rand_temp1] = prop_lpstval1
                
                cur_parm_dict[rand_temp2] = cp.deepcopy(prop_exc2)
                cur_lpval[rand_temp2] = prop_lpval2
                cur_obs[rand_temp2] = prop_obs2
                cur_llval[rand_temp2] = prop_llval2
                cur_lpstval[rand_temp2] = prop_lpstval2
        except (pd.PyDSTool_ExistError, TypeError):
            pyds_error_ctr += 1
    
    #Adapt Proposal Standard Deviations
    if i % 100 == 0 and i < burnin and i <> 0:
        print 'Iteration', i
        update_ctr += 1
        for t in xrange(num_temp):
            for cat in parm_dict.iterkeys():
                for p in parm_dict[cat].iterkeys():
                    try:
                        accept_ratio[t][cat][p].append(accept_prop[t][cat][p]/attempt_prop[t][cat][p])
                        if accept_ratio[t][cat][p][update_ctr] < 0.2:
                            prop_dist[t][cat][p] *= 0.9
                        elif accept_ratio[t][cat][p][update_ctr] < 0.5:
                            pass
                        else:
                            prop_dist[t][cat][p] *= 1.1
                    except ZeroDivisionError:
                        accept_ratio[t][cat][p].append(0)
                        
    # Save Output
    if i > burnin:
        for t in xrange(num_temp):
            for p in parm_dict['pars'].iterkeys():
                par_history[i][t][p] = cur_parm_dict[t]['pars'][p]
            for init in parm_dict['init'].iterkeys():
                init_history[i][t][init] = cur_parm_dict[t]['init'][init]
            for n in parm_dict['noise'].iterkeys():
                noise_history[i][t][n] = cur_parm_dict[t]['noise'][n]
                    
    # Write Output to file on occasion
    if i % 1000 and 1 <> 0:
        sample_dict['pars'] = par_history
        sample_dict['init'] = init_history
        sample_dict['noise'] = noise_history
        save_dict = {'parameter history': sample_dict,
                     'final_prop_dist': prop_dist,
                     'acceptance_rate': accept_ratio,
                     'error_count': pyds_error_ctr}
        save_string = 'pop_mcmc_results.pickle'
        out_file = open(save_string, 'w')
        pickle.dump(sample_dict, out_file)
        out_file.close()

    
#########################################################################
# End Loop
#########################################################################