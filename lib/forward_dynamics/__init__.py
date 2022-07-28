import sympy as sp
from lib.symbols import g, t


class ForwardDynamics:
  def __init__(self, forward_kinematics):
    self.transformations = forward_kinematics.transformations_from_zero_to_i
    self.generalized_coordinates = forward_kinematics.generalized_coordinates

    self._tau = [sp.Symbol(f'tau_{i + 1}') for i in range(len(self.transformations))]
    self._m = [sp.Symbol(f'm_{i + 1}') for i in range(len(self.transformations))]

    total_lagrangian = 0
    for i in range(len(self.transformations)):
      total_lagrangian += self.get_link_lagrangian(i)

    self.total_lagrangian = sp.simplify(total_lagrangian)
    self.equations = self.get_system_equations_of_motion()

  def get_system_equations_of_motion(self):
    equations = []

    for i in range(len(self.transformations)):
      q = self.generalized_coordinates[i]
      dq_dt = sp.diff(q, t)

      eq_1 = sp.Eq(
        sp.diff(sp.diff(self.total_lagrangian, dq_dt), t) - sp.diff(self.total_lagrangian, q),
        self._tau[i]
      )

      equations.append(
        sp.simplify(eq_1)
      )

    return equations

  def get_link_kinetic_energy(self, link_index):
    # Kinetic energy of link i = (1/2) * m_i * (v_i)^2
    m = self._m[link_index]

    v_x = sp.diff(
      self.transformations[link_index][0, 3],
      t
    )

    v_y = sp.diff(
      self.transformations[link_index][1, 3],
      t
    )

    v_z = sp.diff(
      self.transformations[link_index][2, 3],
      t
    )

    return sp.Rational(1, 2) * m * (v_x**2 + v_y**2 + v_z**2)

  def get_link_potential_energy(self, link_index):
    # Potential energy of link i = m_i * g * y_i
    m = self._m[link_index]
    return m * g * self.transformations[link_index][1, 3]

  def get_link_lagrangian(self, link_index):
    return self.get_link_kinetic_energy(link_index) - self.get_link_potential_energy(link_index)

