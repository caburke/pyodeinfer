
import PyDSTool as pd

#########################################################################
# Test Models
#########################################################################

# Fitzhugh Nagumo Model
fhn_ds_args = pd.args()
fhn_ds_args.name = 'Fitzhugh_Nagumo'
fhn_ds_args.fnspecs = {'Jacobian': (['t', 'V', 'R'],
                                    """[[c*(V + pow(V, 2.)), c],
                                    [-1/c, -b/c]]"""),
                            'Vdot': (['V', 'R'], "c*(V - pow(V, 3.)/3. + R)"),
                            'Rdot': (['V', 'R'], "-(V + a - b*R)/c")}
fhn_ds_args.varspecs = {'V': 'Vdot(V, R)',
                        'R': 'Rdot(V, R)'}
                   
fhn_ds_args.ics = {'V': -1., 'R': 1.}
fhn_ds_args.pars = {'a': 0.2, 'b': 0.2, 'c': 3.0}
fhn_ds_args.algparams = {'max_pts': 1000000}
fhn_ds_args.tdata = [0., 20.]

# Goodwin Oscillator (3 components)
goodwin3_args = pd.args()
goodwin3_args.name = 'Goodwin Oscillator (3 Components)'
goodwin3_args.fnspecs = {'Jacobian': (['t', 'X1', 'X2', 'X3'],
                                  """[[ , , ]
                                  [ , , ]
                                  [ , , ]]"""),
                        'X1dot': (['X1', 'X2', 'X3'], "v0/(1 + pow(X3/Km, p) - k1*X1"),
                        'X2dot': (['X1', 'X2', 'X3'], "v1*X1 - k2*X2"),
                        'X3dot': (['X1', 'X2', 'X3'], "v2*X2 - k3*X3")}

#########################################################################
# Darpa Models
#########################################################################

# Model 1 from the pdf
model1_ds_args = pd.args()
model1_ds_args.name = 'Model_1'
model1_ds_args.fnspecs = {'Adot': (['A', 'B'], 'k0 + nu/(1 + pow((B/Ka), m)) - k1*A'),
                          'Bdot': (['A', 'B'], 'k2*A + k3*A*pow(B, n)/(pow(Kb, n) + pow(B, n)) - k4*B')}
model1_ds_args.varspecs = {'A': 'Adot(A, B)',
                           'B': 'Bdot(A, B)'}
model1_ds_args.ics = {'A': 0.0, 'B': 0.0}
model1_ds_args.pars = {'nu': 1.177,
                       'k0': 0.0,
                       'k1': 0.08,
                       'k2': 0.0482,
                       'k3': 1.605,
                       'k4': 0.535,
                       'Ka': 1.1,
                       'Kb': 3.0,
                       'm': 3.0,
                       'n': 2.0}
model1_ds_args.tdata = [0., 200.]
model1_ds_args.algparams = {'max_pts': 1000000}
