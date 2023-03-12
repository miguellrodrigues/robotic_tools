import time

import numpy as np

from lib.inverse_kinematics import *
from robots.comau import comau_fk as fk

np.set_printoptions(suppress=True, precision=6)

# desired real robot position and orientation
desired_transformation = np.array([-415.778, 263.661, 632.794, np.sqrt(5)/2, np.pi/3, -np.pi/4])

start_time = time.time()

thetas, desired_pose, success = evolutive_ik(
    desired_transformation=desired_transformation,
    fk=fk,
    verbose=False,
)

end_time = time.time()

print(' ')
print('Elapsed time:', end_time - start_time)
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