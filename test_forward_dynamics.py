import sympy as sp

from lib.forward_dynamics import ForwardDynamics
from robots.RR import rr_fk as fk

fd = ForwardDynamics(fk)
for eq in fd.equations:
  print(' ')
  sp.print_latex(sp.simplify(eq))
  print(' ')
