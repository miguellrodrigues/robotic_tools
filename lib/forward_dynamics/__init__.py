import sympy as sp
from lib.symbols import g, t


class ForwardDynamics:
  def __init__(self, forward_kinematics):
    self.links = forward_kinematics.links_zero_i
    self.jacobian = forward_kinematics.get_jacobian()

    self.translational_jacobian = self.jacobian[3:, :]
    self.rotational_jacobian    = self.jacobian[:3, :]

    total_lagrangian = 0
    for i in range(len(self.links)):
      total_lagrangian += self.get_link_lagrangian(i)

    self.total_lagrangian = sp.simplify(total_lagrangian)
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

  def get_link_kinetic_energy(self, link_index):
    return self.get_link_translational_kinetic_energy(link_index) + self.get_link_rotational_kinetic_energy(link_index)

  def get_link_translational_kinetic_energy(self, link_index):
    v = self.rotational_jacobian[:, link_index]
    m = self.links[link_index].mass

    return (sp.Rational(1, 2) * m * v.T @ v)[0]

    # v_x = sp.diff(
    #   self.links[link_index].transformation_matrix[0, 3],
    #   t
    # )
    #
    # v_y = sp.diff(
    #   self.links[link_index].transformation_matrix[1, 3],
    #   t
    # )
    #
    # v_z = sp.diff(
    #   self.links[link_index].transformation_matrix[2, 3],
    #   t
    # )
    #
    # return sp.Rational(1, 2) * m * (v_x**2 + v_y**2 + v_z**2)

  def get_link_rotational_kinetic_energy(self, link_index):
    rot_jacobian = self.rotational_jacobian[:, link_index]
    I = self.links[link_index].inertia_tensor

    return (sp.Rational(1, 2) * rot_jacobian.T * I @ rot_jacobian)[0]

  def get_link_potential_energy(self, link_index):
    # Potential energy of link i = m_i * g * y_i
    m = self.links[link_index].mass
    return m * g * self.links[link_index].transformation_matrix[1, 3]

  def get_link_lagrangian(self, link_index):
    return self.get_link_kinetic_energy(link_index) - self.get_link_potential_energy(link_index)

