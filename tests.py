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

j0 = Link([q1, 450,  150, np.pi/2])
j1 = Link([q2,  0,   590, 0])
j2 = Link([q3,  0,   742.07, 0])

fk = ForwardKinematic([j0, j1, j2])

htm = fk.get_homogeneous_transformation_matrix()

P03 = htm[:3, 3]
R03 = sp.lambdify(
  [q1, q2, q3], htm[:3, :3], modules=['numpy']
)

P = sp.lambdify(
  [q1, q2, q3], P03, modules=['numpy']
)

J = sp.lambdify(
  [q1, q2, q3], fk.get_spacial_jacobian(), modules=['numpy']
)

# solving the inverse kinematics for position
theta_i = np.array([.0, .0, .0])
S = desired_position - P(theta_i[0], theta_i[1], theta_i[2])
err = np.linalg.norm(S)

err_tolerance = 1e-3

while err > err_tolerance:
  P_k = P(theta_i[0], theta_i[1], theta_i[2])
  J_k = J(theta_i[0], theta_i[1], theta_i[2])

  theta_i += (np.linalg.pinv(J_k) @ S)[:, 0]

  S = desired_position - P(theta_i[0], theta_i[1], theta_i[2])

  err = np.linalg.norm(S)

print(' ')
print('q1:', theta_i[0])
print('q2:', theta_i[1])
print('q3:', theta_i[2])
print(' ')
print('P:', P(theta_i[0], theta_i[1], theta_i[2])[:, 0])

R36 = R03(theta_i[0], theta_i[1], theta_i[2]).T @ desired_rotation

# # # # # #
q4, q5, q6 = sp.symbols('q_4 q_5 q_6')

_R36 = M([
  [c(q4)*c(q5)*c(q6) - s(q4)*s(q6), -c(q4)*c(q5)*c(q6) - s(q4)*s(q6), c(q4)*s(q5)],
  [s(q4)*c(q5)*c(q6) + c(q4)*s(q6), -s(q4)*c(q5)*c(q6) + c(q4)*s(q6), s(q4)*s(q5)],
  [-s(q5)*c(q6), s(q5)*s(q6), c(q5)],
])

EQS = [
  sp.Eq(R36[i, j], _R36[i, j]) for i in range(3) for j in range(3)
]

# solving for q5
q5_sol = sp.solve(EQS[8], q5)[0]

# solving for q4
q4_sol = sp.solve(EQS[5].subs({q5: q5_sol}), q4)[0]

# solving for q6
q6_sol = sp.solve(EQS[7].subs({q5: q5_sol}), q6)[0]

print(' ')
print('q4:', q4_sol)
print('q5:', q5_sol)
print('q6:', q6_sol)
print(' ')

# # # # # #

