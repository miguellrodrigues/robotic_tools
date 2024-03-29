import time

import numpy as np

from lib.inverse_kinematics import *
from robots.comau import comau_fk as fk

np.set_printoptions(suppress=True, precision=6)

# desired real robot position and orientation
desired_transformation = np.array([498.428, -1008.259, 655.078, 0, 0, 0])

start_time = time.time()

thetas, desired_pose, success = evolutive_ik(
    desired_transformation=desired_transformation,
    fk=fk,
    verbose=True,
)

end_time = time.time()

print(' ')
print('Elapsed time:', end_time - start_time)
print('Success : ', success)
print(' ')

print(' ')
print('Found thetas:', np.rad2deg(thetas))
print(' ')

print('Desired pose:\n', desired_pose)
print(' ')

ee_transformation_matrix = fk.compute_ee_transformation_matrix(thetas)

print('Found pose:\n', ee_transformation_matrix)
print(' ')
print('||(desired_pose - found_pose)||: ', np.linalg.norm(desired_pose - ee_transformation_matrix))
print(' ')