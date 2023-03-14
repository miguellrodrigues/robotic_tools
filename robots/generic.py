# COMAU SMART SiX 6

import numpy as np
import sympy as sp

from lib.forward_kinematics import ForwardKinematic
# from lib.frame import zyz
from lib.link import Link
from lib.symbols import t

np.set_printoptions(suppress=True, precision=5)

q1, q2, q3, q4, q5, q6 = sp.symbols('q_1 q_2 q_3 q_4 q_5 q_6')

j0 = Link([q1, 400, 0, -sp.pi / 2],)
j1 = Link([q2, 0, 550, 0],)
j2 = Link([q3, 0, 300, -sp.pi / 2])
j3 = Link([q4, 400, 0, -sp.pi / 2])
j4 = Link([q5, 0, 0, sp.pi / 2])
j5 = Link([q6, 0, 0, 0])

tcp_angles_offset = np.array([
    0, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2
], dtype=np.float64)

generic_fk = ForwardKinematic(
    [j0, j1, j2, j3, j4, j5],
)
