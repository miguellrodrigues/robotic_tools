from lib.utils import compute_link_transformation


class Link:
  def __init__(
      self,
      dhp=None,
      generalized_coordinate=None,
      link_type='rotational',
      inertia_tensor=None,
      inertia_moment=None,
      mass=None,
      transformation_matrix=None
  ):
    self.dhp = dhp
    self.link_type = link_type
    self.inertia_tensor = inertia_tensor
    self.inertia_moment = inertia_moment
    self.generalized_coordinate = generalized_coordinate

    if generalized_coordinate is None and dhp is None:
      raise ValueError('Either generalized_coordinate or dhp must be specified')

    if dhp is None:
      self.dhp = [generalized_coordinate, 0, 0, 0]

    if generalized_coordinate is None:
      self.generalized_coordinate = dhp[0]

    self.mass = mass

    self.transformation_matrix = transformation_matrix

    if transformation_matrix is None:
      self.transformation_matrix = compute_link_transformation(dhp)

  def get_transformation_matrix(self):
    return self.transformation_matrix
