import sympy as sp

from lib.link import Link
from lib.utils import compute_homogeneous_transformation, cylinder_inertia_tensor


class ForwardKinematic:
  def __init__(self, links):
    self.links = links
    self.len_links = len(self.links)

    self.links_zero_i = []

    for i in range(1, self.len_links + 1):
      m = sp.Symbol(f'm_{i}')

      transformation = self.get_transformation(0, i)
      R = transformation[:3, 3].T

      I = cylinder_inertia_tensor(m)

      self.links_zero_i.append(
        Link(
          generalized_coordinate=self.links[i - 1].dhp[0],
          mass=m,
          transformation_matrix=transformation,
          inertia_tensor=(R @ I @ R.T)[0]
        )
      )

    self.homogeneous_transformation_matrix = sp.simplify(
      self.get_transformation(0, self.len_links)
    )

    self.jacobian = sp.simplify(
      self.get_jacobian()
    )

  def get_transformation(self, start, end):
    tf = compute_homogeneous_transformation(self.links, start, end)
    return tf

  def get_homogeneous_transformation_matrix(self):
    return self.homogeneous_transformation_matrix

  def get_spacial_jacobian(self):
    return self.jacobian[:3, :]

  def get_rotational_jacobian(self):
    return self.jacobian[3:, :]

  def get_jacobian(self):
    htm = self.homogeneous_transformation_matrix

    j = sp.zeros(6, self.len_links)

    # J_pi = Z_i-1 x (P - pi-1)
    # J_oi = z_i-1

    P = htm[:3, 3]

    p_i = sp.Matrix([0, 0, 0])
    z_i = sp.Matrix([0, 0, 1])

    for i in range(1, self.len_links + 1):
      p_diff = (P - p_i)

      J_pi = z_i.cross(p_diff)
      J_oi = z_i

      J = sp.Matrix([J_pi, J_oi])
      j[:, i - 1] = J

      transformation = self.get_transformation(0, i)

      p_i = transformation[:3, 3]
      z_i = transformation[:3, 2]

    return j
