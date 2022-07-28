import sympy as sp
from lib.frame import z_rotation_matrix, translation_matrix, x_rotation_matrix
from lib.symbols import h, r


def compute_link_transformation(dhp):
	rz = z_rotation_matrix(dhp[0])
	tz = translation_matrix(0, 0, dhp[1])
	tx = translation_matrix(dhp[2], 0, 0)
	rx = x_rotation_matrix(dhp[3])
	
	return rz @ tz @ tx @ rx


def compute_homogeneous_transformation(links, start, end):
	if end == 0:
		return sp.eye(4)
	
	transformation_matrix = links[start].get_transformation_matrix()
	
	for i in range(start + 1, end):
		transformation_matrix_i = links[i].get_transformation_matrix()
		transformation_matrix = transformation_matrix @ transformation_matrix_i

	return transformation_matrix


def cylinder_inertia_tensor(m):
	I = sp.Matrix([
		[sp.Rational(1, 12) * m * (3*r**2 + h**2), 0, 0],
		[0, sp.Rational(1, 12)*m*(3*r**2 + h**2), 0],
		[0, 0, sp.Rational(1, 2)*m*r**2]
	])

	return I
