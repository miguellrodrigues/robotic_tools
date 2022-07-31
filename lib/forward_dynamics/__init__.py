import sympy as sp
from lib.symbols import g, t


class ForwardDynamics:
  def __init__(self, forward_kinematics):
    self.jacobian = forward_kinematics.get_jacobian()
    self.links = forward_kinematics.links_zero_i

    self.q = sp.Matrix([link.generalized_coordinate for link in self.links])
    self.dq_dt = self.q.diff(t)
    self.d2q_dt = self.dq_dt.diff(t)

    self.len_q = len(self.q)

    self.w = self.jacobian[3:, :]
    self.equations = self.get_system_equations_of_motion()

  def get_system_equations_of_motion(self):
    equations = []
    D, P = self.get_inertia_matrix_and_potential_energy()

    for k in range(len(self.links)):
      qk = self.q[k]

      tau = sp.Symbol(f'tau_{k + 1}')
      gk = P.diff(qk)

      sum_a = 0
      sum_b = 0

      for i in range(self.len_q):
        qi = self.q[i]

        for j in range(self.len_q):
          qj = self.q[j]
          sum_a += D[k, j] * sp.diff(qj, t, 2)

          dkj = D[k, j]
          dij = D[i, j]

          aux = (dkj.diff(qi) - sp.Rational(1, 2) * dij.diff(qk)) * self.dq_dt[i] * self.dq_dt[j]
          sum_b += aux

      tau_k = sum_a + sum_b + gk

      equations.append(
        sp.Eq(tau, sp.simplify(tau_k))
      )
    
    return equations

  def get_inertia_matrix_and_potential_energy(self):
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

    return D, potential_energy[0]

