# 2R Planar Robot

import sympy as sp

from lib.forward_kinematics import ForwardKinematic
from lib.link import Link
from lib.symbols import t

# To use the forward dynamics, the q's need to be functions of time

q1 = sp.Function('q_1')(t)
q2 = sp.Function('q_2')(t)

a1, a2 = sp.symbols('a_1 a_2')

j0 = Link([q1, 0, a1, 0])
j1 = Link([q2, 0, a2, 0])

rr_fk = ForwardKinematic([j0, j1])
