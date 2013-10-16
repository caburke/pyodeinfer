#########################################################################
# Test Population MCMC Program
#########################################################################

from __future__ import division
import odeinfer.ode_models
import copy as cp
import cPickle as pickle
import math as m
import matplotlib.pyplot as plt
import numpy as np
import PyDSTool as pd
import random as r
import scipy as sp
import scipy.stats

# Define Model
fhn_ds = pd.Generator.Vode_ODEsystem(odeinfer.ode_models.fhn_ds_args)
fhn_traj = fhn_ds.compute('fhn')

# Create Data
obs_times = np.linspace(0, 20, 21)
num_obs = len(obs_times)
true_noise_mean = np.zeros((2, 1))
true_noise_scale = np.array([[0.25], [0.25]])
pure_traj = fhn_traj.sample()
pure_obs = np.array([fhn_traj(obs_times)['V'], fhn_traj(obs_times)['R']])
noisy_obs = pure_obs + np.random.normal(true_noise_mean, true_noise_scale, (2, num_obs))

# Plot Data
#plt.figure(1)
#plt.subplot(211)
#plt.plot(pure_traj['t'], pure_traj['V'], c='b')
#plt.scatter(obs_times, noisy_obs[0,:], c='b')
#plt.axis([0, 20, -3, 3])
#plt.title('Simulated and True Data')
#plt.subplot(212)
#plt.plot(pure_traj['t'], pure_traj['R'], c='r')
#plt.scatter(obs_times, noisy_obs[1,:], c='r')
#plt.axis([0, 20, -2, 2])
#plt.savefig('pop_mcmc_plots/sim_data.pdf')

# Define MCMC Parameters
burnin =0
thin = 1
num_samples= 100
num_iter = burnin + thin*num_samples
num_temp = 11
num_parm = 7
cross_prob = 0.5
temp = np.array([pow(x/(num_temp - 1), 5) for x in range(num_temp - 1)])
temp = np.concatenate([temp, [1.]])
prop_dist0 = {'pars':{'a': sp.stats.norm(0, 0.1), 'b': sp.stats.norm(0, 0.1), 'c': sp.stats.norm(0, 0.1)},
             'init':{'V': sp.stats.norm(0, 0.02), 'R': sp.stats.norm(0, 0.02)},
             'noise':{'V': sp.stats.norm(0, 0.1), 'R': sp.stats.norm(0, 0.1)}}
prop_dist = {}
for i in range(num_temp):
    prop_dist[i] = prop_dist0

# Containers
par_history = np.zeros((num_samples, num_temp), dtype = {'names':['a', 'b', 'c'], 'formats':['float', 'float', 'float']})
init_history = np.zeros((num_samples, num_temp), dtype = {'names':['V', 'R'], 'formats':['float', 'float']})
noise_history = np.zeros((num_samples, num_temp), dtype = {'names':['V', 'R'], 'formats':['float', 'float']})
sample_dict = {'pars': par_history,
               'init': init_history,
               'noise': noise_history}
# Dictionary with starting values for parameters at all temperatures
parm_dict = {'pars': {'a':0.2, 'b':0.2, 'c':3.0}, 
             'init': {'V':-1., 'R':1.}, 
             'noise': {'V':0.5, 'R':0.5}}
cur_parm_dict = {}
for t in range(num_temp):
    cur_parm_dict[t] = {'pars': {'a':0.2, 'b':0.2, 'c':3.0}, 
                         'init': {'V':-1., 'R':1.}, 
                         'noise': {'V':0.5, 'R':0.5}}

# Counters
accept_prop = {}
for t in range(num_temp):
    accept_prop[t] = {'pars':{'a': 0, 'b': 0, 'c': 0},
                     'init':{'V': 0, 'R': 0},
                     'noise':{'V': 0, 'R': 0}}
attempt_prop = cp.deepcopy(accept_prop)
accept_ratio = {}
for t in range(num_temp):
    accept_ratio[t] = {'pars':{'a': [], 'b': [], 'c': []},'init':{'V': [], 'R': []},'noise':{'V': [], 'R': []}}
update_ctr = 0
pyds_error_ctr = 0

#########################################################################
# Probabilistic Model
#########################################################################
# Assign RV's to parameters in model
a = sp.stats.gamma(1., loc = 0., scale = 4.)
b = sp.stats.gamma(1., loc = 0., scale = 4.)
c = sp.stats.gamma(1., loc = 0., scale = 4.)
V0 = sp.stats.norm(-1.0, 0.05)
R0 = sp.stats.norm(1.0, 0.05)
sigmaV = sp.stats.gamma(1., loc = 0., scale = 0.5)
sigmaR = sp.stats.gamma(1., loc = 0., scale = 0.5)

# Dictionary with RV's representing prior distributions
prior_ode_par_dict = {'a':a, 'b':b, 'c':c}
prior_init_dict = {'V': V0, 'R': R0}
prior_noise_dict = {'V': sigmaV, 'R': sigmaR}
prior_dict = {'pars': prior_ode_par_dict,
             'init': prior_init_dict,
             'noise': prior_noise_dict}

# Calculates log prior density given prior dict and current parm values
def log_prior_pdf(prior_dict, parm_dict):
    lpval = 0.0
    for cat in parm_dict.iterkeys():
        for parm in parm_dict[cat].iterkeys():
            lpval += prior_dict[cat][parm].logpdf(parm_dict[cat][parm])
    if np.isfinite(lpval) == False:
        return -1E300
    return lpval
    
# Calculates log likelihood given current parm values and estimated trajectories
def log_likelihood(obs_traj, est_traj, parmdict):
    noise_scale = np.array([[parm_dict['noise']['V']], [parm_dict['noise']['R']]])
    pdf_mat = sp.stats.norm.logpdf(obs_traj, est_traj, noise_scale)
    llval = np.sum(np.sum(pdf_mat))
    if np.isfinite(llval) == False:
        return -1E300
    return llval
    
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
        prop_parm['pars'][p] += prop_dist[rand_temp]['pars'][p].rvs(1)
        prop_lpval = log_prior_pdf(prior_dict, prop_parm)
        
        # Update DS
        fhn_ds.set(pars = prop_parm['pars'], ics = prop_parm['init'])
        try:
            prop_traj = fhn_ds.compute('fhn')
            prop_obs = np.array([prop_traj(obs_times)['V'], prop_traj(obs_times)['R']])
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
     
        except pd.PyDSTool_ExistError:
            pyds_error_ctr += 1
        
   
    # Update Initial Conditions
    for init in parm_dict['init'].iterkeys():
        # Mutate ODE initial conditions and Update Prior
        prop_parm = cp.deepcopy(cur_parm_dict[rand_temp])
        prop_parm['init'][init] += prop_dist[rand_temp]['init'][init].rvs(1)
        prop_lpval = log_prior_pdf(prior_dict, prop_parm)
        
        # Update DS
        fhn_ds.set(pars = prop_parm['pars'], ics = prop_parm['init'])
        try:        
            prop_traj = fhn_ds.compute('fhn')
            prop_obs = np.array([prop_traj(obs_times)['V'], prop_traj(obs_times)['R']])
            
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
        except pd.PyDSTool_ExistError:
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
            fhn_ds.set(pars = prop_cross1['pars'], ics = prop_cross1['init'])
            prop_traj1 = fhn_ds.compute('fhn')
            prop_obs1 = np.array([prop_traj1(obs_times)['V'], prop_traj1(obs_times)['R']])
            prop_llval1 = log_likelihood(noisy_obs, prop_obs1, prop_cross1)
            prop_lpstval1 = prop_lpval1 + temp[rand_temp1]*prop_llval1
            
            prop_lpval2 = log_prior_pdf(prior_dict, prop_cross2)
            fhn_ds.set(pars = prop_cross2['pars'], ics = prop_cross2['init'])
            prop_traj2 = fhn_ds.compute('fhn')
            prop_obs2 = np.array([prop_traj2(obs_times)['V'], prop_traj2(obs_times)['R']])
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
        except pd.PyDSTool_ExistError:
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
            fhn_ds.set(pars = prop_exc1['pars'], ics = prop_exc1['init'])
            prop_traj1 = fhn_ds.compute('fhn')
            prop_obs1 = np.array([prop_traj1(obs_times)['V'], prop_traj1(obs_times)['R']])
            prop_llval1 = log_likelihood(noisy_obs, prop_obs1, prop_exc1)
            prop_lpstval1 = prop_lpval1 + temp[rand_temp1]*prop_llval1
            
            prop_lpval2 = log_prior_pdf(prior_dict, prop_exc2)
            fhn_ds.set(pars = prop_exc2['pars'], ics = prop_exc2['init'])
            prop_traj2 = fhn_ds.compute('fhn')
            prop_obs2 = np.array([prop_traj2(obs_times)['V'], prop_traj2(obs_times)['R']])
            prop_llval2 = log_likelihood(noisy_obs, prop_obs2, prop_exc2)
            prop_lpstval2 = prop_lpval2 + temp[rand_temp1]*prop_llval2
        
            # Determine Accepteance of Exchange Move
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
        except pd.PyDSTool_ExistError:
            pds_error_ctr += 1
    
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
    if i % 100 and 1 <> 0:
        sample_dict['pars'] = par_history
        sample_dict['init'] = init_history
        sample_dict['noise'] = noise_history
        save_string = 'pop_mcmc_results.pickle'
        out_file = open(save_string, 'w')
        pickle.dump(sample_dict, out_file)
        out_file.close()

    
#########################################################################
# End Loop
#########################################################################


