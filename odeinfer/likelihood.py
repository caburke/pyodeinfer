# Likelihoods for ODEinfer

import numpy as np
import scipy.stats as stat

def normal_log_like(obs, states, noise_scale):
    ll_array = stat.distributions.norm.logpdf(obs, states, noise_scale)
    ll_val = np.sum(np.sum(ll_array))
    return ll_val
    
def lognormal_log_like(obs, states, noise_scale):
    ll_array = stat.distributions.lognorm.logpdf(obs, np.exp(states), scale = noise_scale)
    ll_val = np.sum(np.sum(ll_array))
    return ll_val
    
def t_log_like(obs, states, noise_scale, df):
    ll_array = stat.distributions.t.logpdf(obs, df, states, noise_scale)
    ll_val = np.sum(np.sum(ll_array))
    return ll_val
    
def cauchy_log_like(obs, states, noise_scale, df):
    ll_array = stat.distributions.cauchy.logpdf(obs, states, noise_scale)
    ll_val = np.sum(np.sum(ll_array))
    return ll_val