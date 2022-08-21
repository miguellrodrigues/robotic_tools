import numpy as np
from robots.comau import comau_fk as cfk

P1 = np.deg2rad([45, 37, -133, 0, 18, 0])

print(cfk.compute_ee_transformation_matrix(
  cfk.get_angles_from_real_robot(P1)
))

# EE Transformation Matrix
# print(' ')
# sp.print_latex(sp.simplify(fk.ee_transformation_matrix))
# print(' ')
#
# # Jacobian
# sp.print_latex(sp.simplify(fk.jacobian))
# print(' ')
