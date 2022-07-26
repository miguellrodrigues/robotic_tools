import sympy as sp
from lib.constants import g, t


class ForwardDynamics:
  def __init__(self, links):
    self.links = links

    total_lagrangian = 0
    for i in range(len(links)):
      total_lagrangian += self.get_link_lagrangian(i)

    self.total_lagrangian = total_lagrangian
    self.equations = self.get_system_equations_of_motion()

  def get_system_equations_of_motion(self):
    equations = []

    for i in range(len(self.links)):
      theta = self.links[i].dhp[0]
      tau_i = sp.Symbol(f'tau_{i + 1}')

      eq = sp.Eq(
        sp.diff(sp.diff(self.total_lagrangian, sp.diff(theta, t)), t) - sp.diff(self.total_lagrangian, theta),
        tau_i
      )

      equations.append(eq)

    return equations

  def get_link_kinetic_energy(self, link_index):
    # Kinetic energy of link i = 1/2 * m_i * v^2
    m = sp.Symbol(f'm_{link_index}')

    v_x = sp.diff(
      self.links[link_index].get_transformation_matrix()[0, 3],
      t
    )

    v_y = sp.diff(
      self.links[link_index].get_transformation_matrix()[1, 3],
      t
    )

    v_z = sp.diff(
      self.links[link_index].get_transformation_matrix()[2, 3],
      t
    )

    return sp.Rational(1, 2) * m * (v_x**2 + v_y**2 + v_z**2)

  def get_link_potential_energy(self, link_index):
    # Potential energy of link i = m_i * g * y_i
    m = sp.Symbol(f'm_{link_index}')
    return m * g * self.links[link_index].get_transformation_matrix()[2, 3]

  def get_link_lagrangian(self, link_index):
    return self.get_link_kinetic_energy(link_index) - self.get_link_potential_energy(link_index)

