from robots.RR import rr_fk as fk
from lib.forward_dynamics import ForwardDynamics
import sympy as sp


fd = ForwardDynamics(fk)
for eq in fd.equations:
  print(' ')
  sp.print_latex(sp.simplify(eq))
  print(' ')
