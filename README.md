pyodeinfer
==========

Python code for inference in ODE models using Population MCMC. There
is also exploratory code for using Gaussian Process regression to
the time course data, but nothing much came of this.

- Population MCMC http://wwwf.imperial.ac.uk/~das01/Papers/poprev.pdf.

### Dependencies

This code depends on numpy, scipy, PyDSTool, matplotlib, and the
python 2.7 standard library.  The GPy module was used in the 
model1_gpy.py file, but that was an exploration.

### Structure

The populationmcmc_model1.py file runs the Population MCMC algorithm on the 
models, prior distributions, likelihood, and data defined in the odeinfer 
module.  There are also model definitions for the Fitzhugh Nagumo
model which is often used as a simple example for algorithms involving
limit cycle oscillators.  The prior distribution for the model parameters
along with a normal likelihood with mean defined by the solution of a system of ODEs in 
odeinfer/ode_models define the posterior distribution that is the
target of the population MCMC algorithm.

### Results
This repo was a good way to get initial results for this project, but the ODE solver wasn't fast enough
and so the code was migrated to MATLAB to make use of the SUNDIALS ODE Solver toolbox in MATLAB and 
automatic differentiation for the Simplified manifold metropolis adjusted langevin algorithm SMMALA.
This code has a better package structure and can be found in my MATLAB_ODE repo.
