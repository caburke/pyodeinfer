# PyDSTool Specification of Fitzhugh Nagumo ODE

import PyDSTool as pd

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