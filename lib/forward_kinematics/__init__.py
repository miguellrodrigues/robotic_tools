import sympy as sp

from lib.link import Link
from lib.utils import compute_homogeneous_transformation


class ForwardKinematic:
  def __init__(self, links):
    self.links = links
    self.len_links = len(self.links)

    self.links_zero_i = []

    for i in range(1, self.len_links + 1):
      m = sp.Symbol(f'm_{i}')
      # I = sp.symarray(f'I({i})', (3, 3))

      I = sp.Matrix([
        [sp.Symbol(f'I_{i}(xx)'), sp.Symbol(f'I_{i}(xy)'), sp.Symbol(f'I_{i}(xz)')],
        [sp.Symbol(f'I_{i}(xy)'), sp.Symbol(f'I_{i}(yy)'), sp.Symbol(f'I_{i}(yz)')],
        [sp.Symbol(f'I_{i}(xz)'), sp.Symbol(f'I_{i}(yz)'), sp.Symbol(f'I_{i}(zz)')],
      ])

      transformation = self.get_transformation(0, i)
      # I = R @ I @ R.T

      self.links_zero_i.append(
        Link(
          generalized_coordinate=self.links[i - 1].dhp[0],
          mass=m,
          transformation_matrix=sp.simplify(transformation),
          inertia_tensor=I,
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

    # J_vi = Z_i-1 x (P - pi-1)
    # J_wi = z_i-1

    P = htm[:3, 3]

    p_i = sp.zeros(3, 1)
    z_i = sp.Matrix([0, 0, 1])

    for i in range(self.len_links):
      p_diff = (P - p_i)

      J_vi = z_i
      J_wi = sp.zeros(3, 1)

      if self.links[i].link_type == 'R':
        J_vi = z_i.cross(p_diff)
        J_wi = z_i

      J = sp.Matrix([J_vi, J_wi])
      j[:, i] = J

      transformation = self.links_zero_i[i].transformation_matrix

      p_i = transformation[:3, 3]
      z_i = transformation[:3, 2]

    return j
