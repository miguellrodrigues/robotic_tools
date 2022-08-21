import time

import numpy as np

from lib.inverse_kinematics import ik
from robots.comau import comau_fk as fk

np.set_printoptions(suppress=True, precision=5)

desired_transformation = np.array([585.954, -585.954, 250.967, -np.pi/3, -np.pi/6, np.pi/4])

start_time = time.time()

thetas, desired_pose, err = ik(
  desired_transformation=desired_transformation,
  fk=fk,
  lmbd=.1,
  only_position=False,
  verbose=True
)

end_time = time.time()

print(' ')
print('Elapsed time:', end_time - start_time)
print(' ')

offset = np.array([
  0, np.pi/2, -np.pi/2, 0, 0, 0
], dtype=np.float64)

print(' ')
print('Found thetas:', np.rad2deg(
  fk.get_angles_with_offsets(thetas)
))
print('Success:', err)
print(' ')

print('Desired pose:\n', desired_pose)
print(' ')
print('Found pose:\n', fk.compute_ee_transformation_matrix(thetas))
print(' ')
