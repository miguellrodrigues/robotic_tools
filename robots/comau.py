# COMAU SMART SiX 6

import numpy as np
import sympy as sp

from lib.forward_kinematics import ForwardKinematic
from lib.link import Link
from lib.symbols import t

np.set_printoptions(suppress=True, precision=5)

q1, q2, q3, q4, q5, q6 = sp.symbols('q_1 q_2 q_3 q_4 q_5 q_6')

joint_limits = np.deg2rad(
    np.array([
        [-170, 170],
        [-85, 155],
        [0, 170],
        [-210, 210],
        [-130, 130],
        [-270, 270]
    ])
)

j0 = Link([q1, 450, 150, -sp.pi / 2], limits=joint_limits[0])
j1 = Link([q2, 0, 590, sp.pi], limits=joint_limits[1], offset=sp.pi/2)
j2 = Link([q3, 0, 130, -sp.pi / 2], limits=joint_limits[2], offset=-sp.pi/2)
j3 = Link([q4, -647.07, 0, -sp.pi / 2], limits=joint_limits[3])
j4 = Link([q5, 0, 0, sp.pi / 2], limits=joint_limits[4])
j5 = Link([q6, -95, 0, 0], limits=joint_limits[5])

comau_fk = ForwardKinematic(
    [j0, j1, j2, j3, j4, j5],
)
