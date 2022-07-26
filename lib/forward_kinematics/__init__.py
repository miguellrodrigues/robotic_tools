import sympy as sp
from lib.utils import compute_link_transformation, compute_homogeneous_transformation


class Link:
  def __init__(self, dhp):
    self.dhp = dhp
    self.transformation_matrix = compute_link_transformation(dhp)

  def get_transformation_matrix(self):
    return self.transformation_matrix


class ForwardKinematic:
  def __init__(self, links):
    self.links = links
    self.len_links = len(self.links)

    self.transformations_from_zero_to_i = [
      self.get_transformation(0, i) for i in range(1, self.len_links + 1)
    ]

    self.htm = self.get_homogeneous_transformation_matrix()
    self.jacobian = self.get_jacobian()

  def get_transformation(self, start, end):
    tf = compute_homogeneous_transformation(self.links, start, end)
    return tf

  def get_homogeneous_transformation_matrix(self):
    return compute_homogeneous_transformation(self.links, 0, len(self.links))

  def get_spacial_jacobian(self):
    return self.jacobian[:3, :]

  def get_rotational_jacobian(self):
    return self.jacobian[3:, :]

  def get_jacobian(self):
    htm = self.get_homogeneous_transformation_matrix()

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
