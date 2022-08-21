import time

import numpy as np

from lib.inverse_kinematics import ik
from robots.comau import comau_fk as fk

np.set_printoptions(suppress=True, precision=5)

# desired real robot position and orientation
desired_transformation = np.array([-415.778, 863.661, 632.794, 0, 0, 0])

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

print(' ')
print('Found thetas:', np.rad2deg(fk.get_angles_to_real_robot(thetas)))
print('Success:', err)
print(' ')

print('Desired pose:\n', desired_pose)
print(' ')
print('Found pose:\n', fk.compute_ee_transformation_matrix(thetas))
print(' ')
