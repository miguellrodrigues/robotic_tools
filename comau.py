import numpy as np
import sympy as sp

from sympy import cos as c, sin as s, Matrix as M

from lib.forward_kinematics import ForwardKinematic
from lib.link import Link
from lib.frame import x_y_z_rotation_matrix


desired_position = np.array([[892.07], [0], [1170]])
desired_rotation = x_y_z_rotation_matrix(0, 0, 0)[:3, :3]


np.set_printoptions(suppress=True, precision=3)


q1, q2, q3 = sp.symbols('q_1 q_2 q_3')
q4, q5, q6 = sp.symbols('q_4 q_5 q_6')

j0 = Link([q1, 450,  150, np.pi/2])
j1 = Link([q2 + np.pi/2,  0,   590, 0])
j2 = Link([q3,  0,   130, np.pi/2])
j3 = Link([q4,  647.07,   0, np.pi/2])
j4 = Link([q5,  0,   0, -np.pi/2])
j5 = Link([q6,  95,   0, 0])

fk = ForwardKinematic([j0, j1, j2, j3, j4, j5])

htm = fk.get_homogeneous_transformation_matrix()

P06 = htm[:3, 3]

oc = P06 - 95 * np.array([[0], [0], [-1]])

sp.print_latex(sp.simplify(oc))
