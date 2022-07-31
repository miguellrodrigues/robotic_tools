import sympy as sp

from lib.forward_dynamics import ForwardDynamics
from lib.forward_kinematics import ForwardKinematic
from lib.link import Link
from lib.symbols import t

l1, l2, l3 = sp.symbols('a_1 a_2 a_3')

theta1 = sp.Function('q_1')(t)
theta2 = sp.Function('q_2')(t)
theta3 = sp.Function('q_3')(t)

j0 = Link([theta1, 0, l1, 0])
j1 = Link([theta2, 0, l2, 0])

fk = ForwardKinematic([j0, j1])

print(' ')
sp.print_latex(sp.simplify(fk.get_jacobian()))
print(' ')

fd = ForwardDynamics(fk)

for eq in fd.equations:
  print(' ')
  sp.print_latex(eq)
  print(' ')
