#  Copyright (c) Miguel L. Rodrigues 2022.

import matplotlib.pyplot as plt
import numpy as np

from lib.inverse_kinematics import ik
from lib.trajectory import Trajectory
from robots.comau import comau_fk as fk
from robodk.robolink import *

# RDK = Robolink()
# robot = RDK.Item('Comau Smart SiX 6-1.4')

np.set_printoptions(suppress=True, precision=6)

plt.style.use([
  'science',
  'notebook',
  'grid',
])

lmbd_start = .1
lmbd_end = .1

start_transformation = np.array([892.07, .0, 1170, 0, 0, 0])
desired_transformation = np.array([585.95, -585.95, 250.97, -np.pi / 4, 0, 0])

start_time = time.time()

start_angles, _, ss = ik(desired_transformation=start_transformation, fk=fk)
end_angles, _, se = ik(desired_transformation=desired_transformation, fk=fk)

while not (ss and se):
  if not ss:
    lmbd_start -= .1
    start_angles, ss = ik(start_transformation, lmbd=lmbd_start, fk=fk)

  if not se:
    lmbd_end -= .1
    end_angles, se = ik(desired_transformation, lmbd=lmbd_end, fk=fk)

end_time = time.time()

print(' ')
print('Time elapsed:', end_time - start_time)
print(' ')

print(f'Start angles: {np.rad2deg(fk.get_angles_to_real_robot(start_angles))}')
print(f'End angles: {np.rad2deg(fk.get_angles_to_real_robot(end_angles))}')

print(' ')
print('Start transformation:\n', fk.compute_ee_transformation_matrix(start_angles))
print(' ')
print('End transformation:\n', fk.compute_ee_transformation_matrix(end_angles))
print(' ')

start_velocity = np.array([.0, .0, .0, .0, .0, .0])
start_acceleration = np.array([.0, .0, .0, .0, .0, .0])

end_velocity = np.array([.0, .0, .0, .0, .0, .0])
end_acceleration = np.array([.0, .0, .0, .0, .0, .0])

d_t = 10

trajectories = np.empty(6, dtype=Trajectory)
for i in range(len(start_angles)):
  trajectory = Trajectory([start_angles[i], start_velocity[i], start_acceleration[i], end_angles[i], end_velocity[i],
                           end_acceleration[i]], d_t)
  trajectories[i] = trajectory

start_time = 0
end_time = d_t

time_step = 1e-2
iterations = int((end_time - start_time) / time_step)
time_values = np.linspace(start_time, end_time, iterations)

colors = ['r', 'g', 'b', 'y', 'c', 'm']

trajectories_thetas = np.empty((len(trajectories), iterations), dtype=np.float64)
trajectories_velocities = np.empty((len(trajectories), iterations), dtype=np.float64)
trajectories_accelerations = np.empty((len(trajectories), iterations), dtype=np.float64)

for t in range(len(trajectories)):
  t_thetas = np.zeros(iterations)
  t_velocities = np.zeros(iterations)
  t_accelerations = np.zeros(iterations)

  trajectory = trajectories[t]

  for i in range(iterations):
    time = time_values[i]

    t_thetas[i] = trajectory.get_trajectory(time)
    t_velocities[i] = trajectory.get_velocity(time)
    t_accelerations[i] = trajectory.get_acceleration(time)

  trajectories_thetas[t, :] = t_thetas
  trajectories_velocities[t, :] = t_velocities
  trajectories_accelerations[t, :] = t_accelerations

# for i in range(len(time_values)):
#   ths = np.rad2deg(fk.get_angles_to_real_robot(trajectories_thetas[:, i]))
#
#   robot.setJoints(ths.tolist())
#   print(time_values[i])

fig, axs = plt.subplots(3, 1, figsize=(10, 10), tight_layout=True)

for i in range(len(trajectories_thetas)):
  theta_line, = axs[0].plot(time_values, trajectories_thetas[i], color=colors[i])
  vel_line, = axs[1].plot(time_values, trajectories_velocities[i], color=colors[i])
  ac_line, = axs[2].plot(time_values, trajectories_accelerations[i], color=colors[i])

axs[0].set_xlabel('t (s)')
axs[0].set_ylabel(r'$\theta(t)$')

axs[1].set_xlabel('t (s)')
axs[1].set_ylabel(r'$\dot{\theta(t)}$')

axs[2].set_xlabel('t (s)')
axs[2].set_ylabel(r'$\ddot{\theta(t)}$')

plt.savefig('images/trajectories.png', dpi=300)
plt.show()
