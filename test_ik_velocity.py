import time

import numpy as np

from comau import comau_fk as fk
from lib.inverse_velocity_kinematics import ik

np.set_printoptions(suppress=True, precision=6)

desired_transformation = np.array([585.95, -585.95, 250.97, 0, 0, -np.pi/4])

start_time = time.time()

thetas, desired_pose, err = ik(
  desired_transformation=desired_transformation,
  fk=fk,
  verbose=False,
  lmbd=.1,
  only_position=False,
)

end_time = time.time()

print(' ')
print('Elapsed time:', end_time - start_time)
print(' ')

print(' ')
print('Found thetas:', np.rad2deg(thetas + fk.offset))
print('Success:', err)
print(' ')

print('Desired pose:\n', desired_pose)
print(' ')
print('Found pose:\n', fk.compute_ee_transformation_matrix(thetas))
print(' ')
