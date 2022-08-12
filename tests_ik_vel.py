import numpy as np
from lib.inverse_velocity_kinematics import ik
from comau import comau_fk as fk

np.set_printoptions(suppress=True, precision=3)

desired_transformation = np.array([797.07, .0, 1165, 0, 0, 0])

thetas, err = ik(
  desired_transformation=desired_transformation,
  fk=fk,
  verbose=False
)

print(' ')
print('Found thetas:', np.rad2deg(thetas))
print('Success:', err)
print(' ')

print(fk.compute_homogeneous_transformation_matrix(thetas))
