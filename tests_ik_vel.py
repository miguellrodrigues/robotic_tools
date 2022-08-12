import sympy as sp
import numpy as np

from lib.forward_kinematics import ForwardKinematic
from lib.link import Link
from lib.inverse_velocity_kinematics import ik

np.set_printoptions(suppress=True, precision=3)

q1, q2, q3, q4, q5, q6 = sp.symbols('q_1 q_2 q_3 q_4 q_5 q_6')


# change the links here
j0 = Link([q1, 450, 150, np.pi / 2])
j1 = Link([q2, 0, 590, 0])
j2 = Link([q3, 0, 130, np.pi / 2])
j3 = Link([q4, 647.07, 0, np.pi / 2])
j4 = Link([q5, 0, 0, -np.pi / 2])
j5 = Link([q6, 95, 0, 0])

home_offset = np.array([
  0, np.pi/2, 0, 0, -np.pi/2, np.pi
], dtype=np.float64)

fk = ForwardKinematic([j0, j1, j2, j3, j4, j5])

print(fk.compute_homogeneous_transformation_matrix(home_offset))

desired_transformation = np.array([797.07, .0, 1165, 0, np.pi/4, 0])

thetas, err = ik(
  desired_transformation=desired_transformation,
  fk=fk,
  home_offset=home_offset,
  verbose=False
)

print(' ')
print('Found theta:', np.rad2deg(thetas))
print('Success:', err)
print(' ')

print(fk.compute_homogeneous_transformation_matrix(thetas))
