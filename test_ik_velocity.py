import numpy as np
from lib.inverse_velocity_kinematics import ik
from comau import comau_fk as fk

np.set_printoptions(suppress=True, precision=6)

desired_transformation = np.array([797.07, .0, 1075, np.pi/6, np.pi/6, 0])

thetas, desired_pose, err = ik(
  desired_transformation=desired_transformation,
  fk=fk,
  verbose=False,
  lmbd=.1
)

print(' ')
print('Found thetas:', np.rad2deg(thetas + fk.offset))
print('Success:', err)
print(' ')

print('Desired pose:\n', desired_pose)
print(' ')
print('Found pose:\n', fk.compute_homogeneous_transformation_matrix(thetas))
print(' ')
