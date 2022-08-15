# COMAU SMART SiX 6

import numpy as np
import sympy as sp

from lib.forward_kinematics import ForwardKinematic
from lib.link import Link

q1, q2, q3, q4, q5, q6 = sp.symbols('q_1 q_2 q_3 q_4 q_5 q_6')

j0 = Link([q1, 450, 150, sp.pi / 2])
j1 = Link([q2, 0, 590, 0])
j2 = Link([q3, 0, 130, sp.pi / 2])
j3 = Link([q4, 647.07, 0, -sp.pi / 2])
j4 = Link([q5, 0, 0, sp.pi / 2])
j5 = Link([q6, 95, 0, 0])

home_offset = np.array([
  0, np.pi / 2, np.pi / 2, 0, 0, np.pi
], dtype=np.float64)

comau_fk = ForwardKinematic([j0, j1, j2, j3, j4, j5], offset=home_offset)
