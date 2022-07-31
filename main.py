import sympy as sp

from lib.forward_dynamics import ForwardDynamics
from lib.forward_kinematics import ForwardKinematic
from lib.link import Link
from lib.symbols import t

a1, a2 = sp.symbols('a_1 a_2')

q1 = sp.Function('q_1')(t)
q2 = sp.Function('q_2')(t)

j0 = Link([q1, 0, a1, sp.pi/2])
j1 = Link([q2, 0, a2, 0])

fk = ForwardKinematic([j0, j1])

print(' ')
sp.print_latex(sp.simplify(fk.get_jacobian()))
print(' ')

fd = ForwardDynamics(fk)

for eq in fd.equations:
  print(' ')
  sp.print_latex(eq)
  print(' ')
