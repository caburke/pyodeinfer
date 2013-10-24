# Development of Gaussian Process inference on Model 1

# Imports
import cPickle
import GPy as gp
import matplotlib.pyplot as plt
import numpy as np

# Load Data
data_file = open('/home/chris/python/pyodeinfer/odeinfer/data/data_4_genes.pkl', 'r')
data_array1 = cPickle.load(data_file)
data_file.close()
data_array2 = data_array1[range(10, 123)]  # Remove transient phase
frq_bad_orig = data_array2['x661_1']  # Garbage data
frq_orig = data_array2['x661_4']
stk_orig = data_array2['stk']
clb_orig = data_array2['clb']
cln_orig = data_array2['cln']

# Transform Raw Data
def transform_data(data_vector):
    data_min = min(np.min(frq_orig), np.min(stk_orig), np.min(clb_orig), np.min(cln_orig))
    data_max = max(np.max(frq_orig), np.max(stk_orig), np.max(clb_orig), np.max(cln_orig))
    new_data = (data_vector - data_min) / (data_max - data_min)
    return new_data

def GPy_data_format(data_vector):
    gpy_data = np.reshape(data_vector, (len(data_vector), 1))
    return gpy_data

# Transformed Data
frq = transform_data(frq_orig)
stk = transform_data(stk_orig)
clb = transform_data(clb_orig)
cln = transform_data(cln_orig)
N = len(frq)

# GPy data
frq_g = np.reshape(frq, (N, 1))
stk_g = np.reshape(stk, (N, 1))
clb_g = np.reshape(clb, (N, 1))
cln_g = np.reshape(cln, (N, 1))
t = np.linspace(0, N - 1, N)
t_g = np.reshape(t, (N, 1))

# Plot Raw Data
t = np.arange(len(clb))
plt.plot(t, frq, label='frq')
plt.plot(t, stk, label='stk')
plt.plot(t, clb, label='clb')
plt.plot(t, cln, label='cln')
plt.legend(loc=4)

###########################################################################
# GPy Code
###########################################################################

# Trend model
kernel_trend = gp.kern.Matern52(1)
model_trend = gp.models.GPRegression(t_g, frq_g, kernel_trend)
model_trend.unconstrain('')
model_trend.constrain_positive('.*Mat52_variance')
model_trend.constrain_positive('.*noise_variance')
model_trend.constrain_fixed('.*Mat52_lengthscale', 20)
model_trend.optimize_restarts(10)
#model_trend.plot()

# Full Model
kernel_per = gp.kern.periodic_Matern52(1)
kernel_ap = gp.kern.rbf(1)
kernel_tot = kernel_per*kernel_ap

model_tot = gp.models.GPRegression(t_g, frq_g, kernel_tot)
model_tot.unconstrain('')
model_tot.constrain_bounded('.*periodic_Mat52_variance', 0.1, 1.0)
model_tot.constrain_bounded('.*periodic_Mat52_lengthscale', 15., 25.)
model_tot.constrain_bounded('.*periodic_Mat52_period', 21, 23)
model_tot.constrain_bounded('.*rbf_variance', 0.1, 1.0)
model_tot.constrain_fixed('.*rbf_lengthscale', 250.)
model_tot.constrain_positive('.*noise_variance')
model_tot.optimize_restarts(num_restarts = 5)

# Code to Generate Sample Paths
X = np.linspace(0, 120, num=1201)[:, None]
Y_per = np.random.multivariate_normal(np.zeros(len(X)), kernel_per.K(X))
plt.plot(X, Y_per)
Y_ap = np.random.multivariate_normal(np.zeros(len(X)), kernel_ap.K(X))
plt.plot(X, Y_ap)
Y_tot = np.random.multivariate_normal(np.zeros(len(X)), kernel_tot.K(X))
plt.plot(X, Y_tot)

# Gaussian Process Analysis
hyper_parm_dict = {'lamda_per': 1.,
                   'nu_per': 2.5,
                   'sigma2_per': 1.,
                   'theta_per': 1.,
                   'nu_ap': 2.5,
                   'sigma2_ap': 1.,
                   'theta_ap': 1.}
