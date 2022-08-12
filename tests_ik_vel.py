import numpy as np
from lib.inverse_velocity_kinematics import ik
from comau import comau_fk as fk

np.set_printoptions(suppress=True, precision=6)

desired_transformation = np.array([497.07, .0, 1165, np.pi/4, -np.pi/4, -np.pi/6])

thetas, err = ik(
  desired_transformation=desired_transformation,
  fk=fk,
  verbose=True
)

print(' ')
print('Found thetas:', np.rad2deg(thetas))
print('Success:', err)
print(' ')

print(fk.compute_homogeneous_transformation_matrix(thetas))
