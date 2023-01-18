import sympy as sp

from lib.forward_dynamics import ForwardDynamics
from robots.comau import comau_fk as fk

fd = ForwardDynamics(fk)
for eq in fd.equations:
    print(' ')
    sp.print_latex(sp.simplify(eq))
    print(' ')
