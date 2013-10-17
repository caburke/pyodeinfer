# Custom functions for plotting MCMC output

import matplotlib.pyplot as plt

# Plots histogram of chosen parameter chain
def parameter_histogram(parm_hist, parm_cat, parm_name, bins = 10,
                        show = True, save_file = 'temp_hist', 
                        file_type = 'png', temp = None):
    if temp == None:
        # Not population MCMC
        plt.hist(parm_hist[parm_cat][parm_name][:], bins)
        if show == True:
            plt.show()
        else:
            plt.savefig(save_file + file_type, format = file_type)
    else:
        # Population MCMC
        plt.hist(parm_hist[parm_cat][parm_name][:, temp], bins)
        if show == True:
            plt.show()
        else:
            plt.savefig(save_file + file_type, format = file_type)
            
# Plots trace of chosen parameter chain
def parameter_trace(parm_hist, parm_cat, parm_name,
                        show = True, save_file = 'temp_trace', 
                        file_type = 'png', temp = None)