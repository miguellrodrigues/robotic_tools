import sympy as sp
from lib.symbols import g, t


class ForwardDynamics:
  def __init__(self, forward_kinematics):
    self.jacobian = forward_kinematics.get_jacobian()
    self.links = forward_kinematics.links_zero_i

    self.q = sp.Matrix([link.generalized_coordinate for link in self.links])
    self.dq_dt = self.q.diff(t)

    self.len_q = len(self.q)

    self.w = self.jacobian[3:, :]

    k, p = self.get_total_energy()

    self.total_lagrangian = k - p
    self.equations = self.get_system_equations_of_motion()

  def get_system_equations_of_motion(self):
    equations = []

    for i in range(len(self.links)):
      q = self.q[i]
      dq_dt = self.dq_dt[i]
      tau = sp.Symbol(f'tau_{i + 1}')

      EQ = sp.diff(sp.diff(self.total_lagrangian, dq_dt), t) - sp.diff(self.total_lagrangian, q)

      equations.append(
        sp.Eq(tau, sp.simplify(EQ[0]))
      )

    return equations

  def get_total_energy(self):
    # translational_kinetic_energy = 0
    potential_energy = sp.zeros(1, 1)
    D = sp.zeros(self.len_q, self.len_q)
    G = sp.Matrix([0, -g, 0])

    for i in range(len(self.links)):
      m = self.links[i].mass
      I = self.links[i].inertia_tensor

      Jwi = sp.zeros(3, len(self.q))
      Jvi = sp.zeros(3, len(self.q))

      Jwi[:, :i + 1] = self.w[:, :i + 1]

      r = self.links[i].transformation_matrix[:3, 3]
      dr_dq = [sp.diff(r, q) for q in self.q]

      for j in range(self.len_q):
        Jvi[:, j] = dr_dq[j]

      D += (m * Jvi.T @ Jvi) + (Jwi.T @ I @ Jwi)
      potential_energy += m * G.T @ r

    K = sp.Rational(1, 2) * self.dq_dt.T @ D @ self.dq_dt

    return K, potential_energy

