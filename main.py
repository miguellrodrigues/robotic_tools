from lib.forward_kinematics import Link, ForwardKinematic
import sympy as sp
from lib.forward_dynamics import ForwardDynamics
from lib.constants import t

l2, l3, l4, l5, d3 = sp.symbols('l2 l3 l4 l5 d3')

theta1 = sp.Function('theta_1')(t)
theta2 = sp.Function('theta_2')(t)
theta3 = sp.Function('theta_3')(t)
theta4 = sp.Function('theta_4')(t)

j0 = Link([theta1, 0, l2, 0])
j1 = Link([theta2, 0, l3, 0])

fk = ForwardKinematic([j0, j1])
fd = ForwardDynamics(fk)

sp.print_latex(
  fd.equations[0]
)

