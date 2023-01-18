import sympy as sp
from lib.forward_dynamics import ForwardDynamics
from robots.comau import comau_fk as fk

fd = ForwardDynamics(fk)
i = 0

for eq in fd.equations:
    simplified = sp.simplify(eq)

    print(' ')
    sp.pprint(simplified.args)
    print(' ')
