import sympy as sp
from robots.RR import rr_fk as fk


# EE Transformation Matrix
print(' ')
sp.print_latex(sp.simplify(fk.ee_transformation_matrix))
print(' ')

# Jacobian
sp.print_latex(sp.simplify(fk.jacobian))
print(' ')
