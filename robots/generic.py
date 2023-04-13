import numpy as np
import sympy as sp

from lib.forward_kinematics import ForwardKinematic
from lib.link import Link
from lib.symbols import t

np.set_printoptions(suppress=True, precision=5)

q1, q2, q4, q5, q6 = sp.symbols('q_1 q_2 q_4 q_5 q_6')
d3 = sp.symbols('d_3')

d2 = .164
d4 = .05
d5 = .143
d6 = .12

j0 = Link([q1, 0,      0,     -sp.pi / 2])
j1 = Link([q2, 0,      0,      sp.pi / 2], offset=sp.pi/2)
j2 = Link([0,  d3,     0,              0], link_type='P')
j3 = Link([q4, d2+d4,  0,      sp.pi / 2], offset=-sp.pi/2)
j4 = Link([q5, 0,      0,     -sp.pi / 2])
j5 = Link([q6, d6+d5,      0,             0])

generic_fk = ForwardKinematic(
    [j0, j1, j2, j3, j4, j5],
)
