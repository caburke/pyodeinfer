# Priors for models in ode_models

import numpy as np
import scipy as sp
import scipy.stats

def log_prior_pdf(prior_dict, parm_dict):
    lpval = 0.0
    for cat in parm_dict.iterkeys():
        for parm in parm_dict[cat].iterkeys():
            lpval += prior_dict[cat][parm].logpdf(parm_dict[cat][parm])
    if np.isfinite(lpval) == False:
        return -1E300
    return lpval

fhn_ds_prior = {'pars':{'a': sp.stats.gamma(1, loc=0, scale=4),
                        'b': sp.stats.gamma(1, loc=0, scale=4),
                        'c': sp.stats.gamma(1, loc=0, scale=4)},
                    'init':{'V0': sp.stats.norm(-1.0, 0.1),
                            'R0': sp.stats.norm(1.0, 0.1)},
                    'noise':{'sigmaV': sp.stats.gamma(1, loc=0, scale=0.5),
                             'sigmaR': sp.stats.gamma(1, loc=0, scale=0.5)}}

model1_ds_prior = {'pars':{'nu': sp.stats.gamma(1, loc=0, scale=4),
                           'k0': sp.stats.gamma(1, loc=0, scale=4),
                           'k1': sp.stats.gamma(1, loc=0, scale=4),
                           'k2': sp.stats.gamma(1, loc=0, scale=4),
                           'k3': sp.stats.gamma(1, loc=0, scale=4),
                           'k4': sp.stats.gamma(1, loc=0, scale=4),
                           'Ka': sp.stats.gamma(1, loc=0, scale=4),
                           'Kb': sp.stats.gamma(1, loc=0, scale=4)},
                    'init':{'A': sp.stats.norm(loc=1.639,scale= 0.1),
                            'B': sp.stats.norm(loc=1.433,scale= 0.1)},
                   'noise':{'A': sp.stats.gamma(1, loc=0, scale=0.2),
                            'B': sp.stats.gamma(1, loc=0, scale=0.2)}}