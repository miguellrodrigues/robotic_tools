import numpy as np
import sympy as sp

from sympy import cos as c, sin as s, Matrix as M

from lib.forward_kinematics import ForwardKinematic
from lib.link import Link


desired_position = np.array([[797.07], [0], [1170]])
desired_rotation = np.array([
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1]
])  # x_y_z_rotation_matrix(0, 0, 0)[:3, :3]


np.set_printoptions(suppress=True, precision=3)


q1, q2, q3 = sp.symbols('q_1 q_2 q_3')

j0 = Link([q1, 450,  150, np.pi/2])
j1 = Link([q2,  0,   720, 0])
j2 = Link([q3,  0,   647.07, -np.pi/2])

home_offset = np.array([0, np.pi/2, -np.pi/2])
fk = ForwardKinematic([j0, j1, j2], offset=home_offset)

htm = fk.get_homogeneous_transformation_matrix()

P03 = htm[:3, 3]
R03 = sp.lambdify(
  [(q1, q2, q3)], htm[:3, :3], modules=['numpy']
)

P = sp.lambdify(
  [(q1, q2, q3)], P03, modules=['numpy']
)

J = sp.lambdify(
  [(q1, q2, q3)], fk.get_spacial_jacobian(), modules=['numpy']
)

# solving the inverse kinematics for position
theta_k = np.array([[.0], [.0], [.0]])
err_tolerance = 1e-6
F = err_tolerance + 1
gamma = .1

while F > err_tolerance:
  P_k = P(theta_k[:, 0] + home_offset)

  # associated function G(theta) = P(theta) - P_d
  G = P_k - desired_position

  # objective function F(theta) = 1/2 G(theta)^T G(theta)
  F = .5 * G.T @ G

  # gradient of G(theta) 'Jacobian'
  J_k = J(theta_k[:, 0] + home_offset)

  theta_k_1 = theta_k - gamma * (J_k.T @ G)
  theta_k = theta_k_1


theta_i = theta_k[:, 0] + home_offset
oc = P(theta_i)[:, 0]

print(' ')
print('q1:', np.rad2deg(theta_i[0]))
print('q2:', np.rad2deg(theta_i[1]))
print('q3:', np.rad2deg(theta_i[2]))
print(' ')
print('Oc:', oc)

R36 = R03(theta_i).T @ desired_rotation

# # # # # #
q4, q5, q6 = sp.symbols('q_4 q_5 q_6')

_R36 = M([
  [c(q4)*c(q5)*c(q6) - s(q4)*s(q6), -c(q4)*c(q5)*c(q6) - s(q4)*c(q6), c(q4)*s(q5)],
  [s(q4)*c(q5)*c(q6) + c(q4)*s(q6), -s(q4)*c(q5)*s(q6) + c(q4)*c(q6), s(q4)*s(q5)],
  [-s(q5)*c(q6), s(q5)*s(q6), c(q5)],
])

EQS = [
  sp.Eq(c(q5 + np.pi/2), R36[2, 2]),
  sp.Eq(-s(q4)*s(q5 + np.pi/2), R36[1, 2]),
  sp.Eq(-s(q5 + np.pi/2)*s(q6 - np.pi/2), R36[2, 1]),
]

# solving for q5
q5_sol = sp.solve(EQS[0], q5)[0]

# solving for q4
q4_sol = sp.solve(EQS[1].subs({q5: q5_sol}), q4)[0]

# solving for q6
q6_sol = sp.solve(EQS[2].subs({q5: q5_sol}), q6)[0]

print(' ')
print('q4:', np.rad2deg(float(q4_sol)))
print('q5:', np.rad2deg(float(q5_sol)))
print('q6:', np.rad2deg(float(q6_sol)))
print(' ')

# # # # # #

ee_pos = oc + (95 * desired_rotation @ np.array([0, 0, 1]))

print('EE position:', ee_pos)
print(' ')
