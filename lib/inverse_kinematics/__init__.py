#  Copyright (c) Miguel L. Rodrigues 2022.

import numpy as np
import sympy as sp

from lib.forward_kinematics import ForwardKinematic
from lib.frame import x_y_z_rotation_matrix, translation_matrix
from lib.utils import matrix_log6, inverse_transformation, se3_to_vec, normalize_angle_between_limits
from scipy.optimize import differential_evolution


def de_ik(
        desired_transformation=None,
        fk: ForwardKinematic = None,
        initial_guess=None,
        max_iterations=1000,
        verbose=False,
):
    if initial_guess is None:
        initial_guess = np.random.uniform(0, 2 * np.pi, fk.len_links)

    desired_rotation = x_y_z_rotation_matrix(desired_transformation[3], desired_transformation[4],
                                             desired_transformation[5])

    desired_pose = sp.matrix2numpy(translation_matrix(desired_transformation[0], desired_transformation[1],
                                                      desired_transformation[2]) @ desired_rotation, dtype=np.float64)

    theta_i = initial_guess.copy()

    error = True
    i = 0

    def cost_function(thetas):
        htm = fk.compute_ee_transformation_matrix(thetas)
        i_htm = inverse_transformation(htm)

        T_bd = i_htm @ desired_pose
        log_tbd = matrix_log6(T_bd)

        s = se3_to_vec(log_tbd)
        n_s = np.linalg.norm(s)

        if verbose:
            print(f'Iteration {i}, error = {n_s}')

        return n_s

    bounds = [(link.limits[0], link.limits[1]) for link in fk.links]

    res = differential_evolution(cost_function, bounds, maxiter=max_iterations)

    if res.success:
        theta_i = res.x
        error = False
    
    return theta_i, desired_pose, not error


def ik_position(
        desired_position=None,
        fk: ForwardKinematic = None,
        initial_guess=None,
        f_tolerance=1e-7,
        max_iterations=1500,
        lmbd=.1,
        verbose=False,
        normalize=False
):
    desired_position = np.array([
        [desired_position[0]],
        [desired_position[1]],
        [desired_position[2]]
    ])

    if initial_guess is None:
        initial_guess = np.random.uniform(0, 2 * np.pi, fk.len_links)

    theta_i = initial_guess.copy()

    F = f_tolerance + 1
    i = 0

    while F > f_tolerance and i < max_iterations:
        P_i = fk.compute_ee_position(theta_i)
        G = P_i - desired_position

        F = .5 * G.T @ G

        J_k = fk.compute_jacobian(theta_i)[3:, :]

        theta_i -= lmbd * (np.linalg.pinv(J_k) @ G)[:, 0]

        i += 1

        if verbose:
            print(f'Iteration {i}, F = {F}')

    error = F > f_tolerance

    # Use theta_i + fk.offset in external applications
    # For calculations using this lib, use theta_i with the offset

    if normalize:
        for i in range(fk.len_links):
            link_limits = fk.links[i].limits

            theta = normalize_angle_between_limits(
                theta_i[i],
                link_limits[0],
                link_limits[1]
            )

            theta_i[i] = theta

    return theta_i, desired_position, not error


def ik(
        desired_transformation=None,
        fk: ForwardKinematic = None,
        initial_guess=None,
        epsilon_wb=1e-5,
        epsilon_vb=1e-5,
        max_iterations=1000,
        lmbd=.1,
        verbose=False,
        only_position=False,
        normalize=True):
    # finding the thetas only for the position
    if only_position:
        return ik_position(
            desired_position=desired_transformation[:3],
            fk=fk,
            initial_guess=initial_guess,
            f_tolerance=epsilon_vb,
            max_iterations=max_iterations,
            lmbd=lmbd,
            verbose=not verbose)

    # transformation_data = [x, y, z, rx, ry, rz]
    # x, y, z: position of the end effector
    # rx, ry, rz: orientation of the end effector
    # returns: the joint angles

    # The end effector z-axis must be in the same direction and sign as the z-axis of the base frame z-axis

    if initial_guess is None:
        initial_guess = np.random.uniform(0, 2 * np.pi, fk.len_links)

    desired_rotation = x_y_z_rotation_matrix(desired_transformation[3], desired_transformation[4],
                                             desired_transformation[5])

    desired_pose = sp.matrix2numpy(translation_matrix(desired_transformation[0], desired_transformation[1],
                                                      desired_transformation[2]) @ desired_rotation, dtype=np.float64)

    theta_i = initial_guess.copy()

    error = True
    i = 0

    while error and i < max_iterations:
        htm = fk.compute_ee_transformation_matrix(theta_i)
        i_htm = inverse_transformation(htm)

        T_bd = i_htm @ desired_pose
        log_tbd = matrix_log6(T_bd)

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
    # For calculations using this lib, use theta_i with the offset

    if error:
        theta_pos, _, success_pos = ik_position(
            desired_position=desired_transformation[:3],
            fk=fk,
            initial_guess=initial_guess,
            f_tolerance=epsilon_vb,
            max_iterations=max_iterations,
            lmbd=lmbd,
            verbose=not verbose
        )

        if success_pos:
            theta_i = theta_pos

    if normalize:
        for i in range(fk.len_links):
            link_limits = fk.links[i].limits

            theta = normalize_angle_between_limits(
                theta_i[i],
                link_limits[0],
                link_limits[1]
            )

            theta_i[i] = theta

    return theta_i, desired_pose, 'Full' if not error else 'Partial' if success_pos else 'None'
