import sympy as sp
from lib.symbols import g, t


class ForwardDynamics:
  def __init__(self, forward_kinematics):
    self.links = forward_kinematics.links_zero_i
    self.jacobian = forward_kinematics.get_jacobian()

    self.q = sp.Matrix([link.generalized_coordinate for link in self.links])
    self.dq_dt = self.q.diff(t)

    self.translational_jacobian = self.jacobian[:3, :]
    self.rotational_jacobian = self.jacobian[3:, ::-1]

    self.total_lagrangian = self.get_total_kinetic_energy() - self.get_total_potential_energy()
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
    translational_kinetic_energy = 0
    rotational_kinetic_energy = 0

    for i in range(len(self.links)):
      m = self.links[i].mass
      I = self.links[i].inertia_tensor

      if i == 0 and len(self.dq_dt) >= 2:
        w = sp.Matrix([
          [self.rotational_jacobian[:, :i + 1], sp.zeros(3, 1)],
        ]) @ self.dq_dt
      else:
        w = (self.rotational_jacobian[:, :i + 1] @ self.dq_dt)

      r = self.links[i].transformation_matrix[:3, 3]
      v = r.diff(t)

      translational_kinetic_energy += (sp.Rational(1, 2) * m * (v.T @ v))[0]
      rotational_kinetic_energy += (sp.Rational(1, 2) * (w.T @ I @ w))[0]

    return translational_kinetic_energy + rotational_kinetic_energy

  def get_total_potential_energy(self):
    P = 0

    for i in range(len(self.links)):
      m = self.links[i].mass
      P += m * g * self.links[i].transformation_matrix[1, 3]

    return P

