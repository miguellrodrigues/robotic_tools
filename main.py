from lib.forward_kinematics import Link, DirectKinematic
import sympy as sp

l2, l3, l4, l5, d3 = sp.symbols('l2 l3 l4 l5 d3')
theta1, theta2, theta3, theta4 = sp.symbols('theta_1 theta_2 theta_3 theta_4')

j0 = Link([theta1, 0, l2, 0])
j1 = Link([theta2, 0, l3, sp.pi])
j2 = Link([theta3, l4 + d3, 0, 0])
j3 = Link([theta4, l5, 0, 0])

dk = DirectKinematic([j0, j1, j2, j3])

sp.print_latex(dk.get_spacial_jacobian())
