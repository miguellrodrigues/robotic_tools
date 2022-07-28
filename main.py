import numpy as np
import sympy as sp

from lib.forward_dynamics import ForwardDynamics
from lib.forward_kinematics import Link, ForwardKinematic
from lib.symbols import t

l1, l2 = sp.symbols('l1 l2')

theta1 = sp.Function('theta_1')(t)
theta2 = sp.Function('theta_2')(t)

j0 = Link([theta1, 0, l1, 0])
j1 = Link([theta2, 0, l2, 0])

fk = ForwardKinematic([j0, j1])
fd = ForwardDynamics(fk)

print(' ')
sp.print_latex(sp.simplify(fk.get_homogeneous_transformation_matrix()))
print(' ')