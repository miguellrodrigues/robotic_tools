#  Copyright (c) Miguel L. Rodrigues 2022.

import numpy as np
import sympy as sp

from lib.forward_kinematics import ForwardKinematic
from lib.frame import x_y_z_rotation_matrix, translation_matrix
from lib.utils import matrix_log6, inverse_transformation, se3_to_vec


def ik(
  desired_transformation=None,
  fk: ForwardKinematic = None,
  initial_guess=None,
  epsilon_wb=1e-7,
  epsilon_vb=1e-7,
  max_iterations=1500,
  lmbd=.1,
  verbose=False):
  # transformation_data = [x, y, z, rx, ry, rz]
  # x, y, z: position of the end effector
  # rx, ry, rz: orientation of the end effector
  # returns: the joint angles

  # The end effector z-axis must be in the same direction and sign as the z-axis of the base frame

  n = fk.len_links

  if initial_guess is None:
    initial_guess = np.array([.0 for _ in range(n)])

  desired_rotation = x_y_z_rotation_matrix(desired_transformation[3], desired_transformation[4],
                                           desired_transformation[5])

  desired_pose = sp.matrix2numpy(translation_matrix(desired_transformation[0], desired_transformation[1],
                                                    desired_transformation[2]) @ desired_rotation, dtype=np.float64)

  theta_i = initial_guess

  error = True
  i = 0

  while error and i < max_iterations:
    htm = fk.compute_homogeneous_transformation_matrix(theta_i)
    i_htm = inverse_transformation(htm)

    Tbd = i_htm @ desired_pose
    log_tbd = matrix_log6(Tbd)

    s = se3_to_vec(log_tbd)

    J = fk.compute_jacobian(theta_i)

    d_theta = np.linalg.pinv(J) @ s
    theta_i += (lmbd * d_theta)

    wb_err = np.linalg.norm(s[:3])
    vb_err = np.linalg.norm(s[3:])

    error = wb_err > epsilon_wb or vb_err > epsilon_vb

    i += 1

    if verbose:
      print(f'Iteration {i}, s = {s}')

  # Use theta_i + fk.offset in external applications
  # For calculations using this bib, use theta_i with the offset
  return theta_i, desired_pose, not error
