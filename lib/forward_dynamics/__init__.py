import sympy as sp
from lib.symbols import g, t


class ForwardDynamics:
  def __init__(self, forward_kinematics):
    self.links = forward_kinematics.links_zero_i
    self.jacobian = forward_kinematics.get_jacobian()

    self.translational_jacobian = self.jacobian[3:, :]
    self.rotational_jacobian    = self.jacobian[:3, :]


    # # # # # # # # # # # # # # #


    D = self.get_total_kinetic_energy()
    P = self.get_total_potential_energy()

    q = [self.links[i].generalized_coordinate for i in range(len(self.links))]
    dq = sp.Matrix([sp.diff(q[i], t) for i in range(len(self.links))])

    self.K = (sp.Rational(1, 2) * dq.T * D @ dq)[0]
    self.P = P

    self.total_lagrangian = self.K - self.P
    self.equations = self.get_system_equations_of_motion()

  def get_system_equations_of_motion(self):
    equations = []

    for i in range(len(self.links)):
      q = self.links[i].generalized_coordinate
      dq_dt = sp.diff(q, t)

      tau = sp.Symbol(f'tau_{i + 1}')

      eq_1 = sp.Eq(
        sp.diff(sp.diff(self.total_lagrangian, dq_dt), t) - sp.diff(self.total_lagrangian, q),
        tau
      )

      equations.append(
        sp.simplify(eq_1)
      )

    return equations

  def get_total_kinetic_energy(self):
    D = 0

    for i in range(len(self.links)):
      m = self.links[i].mass
      I = self.links[i].inertia_tensor

      trans_jacobian = self.translational_jacobian[:, i]
      rot_jacobian = self.rotational_jacobian[:, i]

      D += (m * trans_jacobian.T @ trans_jacobian + rot_jacobian.T * I @ rot_jacobian)[0]

    return D

  def get_total_potential_energy(self):
    P = 0

    for i in range(len(self.links)):
      m = self.links[i].mass
      P += m * g * self.links[i].transformation_matrix[1, 3]

    return P

