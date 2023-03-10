import sympy as sp
from lib.forward_dynamics import ForwardDynamics
from robots.comau import comau_fk as fk

fd = ForwardDynamics(fk)
i = 0

print(1)

D, C, G = fd.D, fd.C, fd.G

matrices = [D,C,G]
names    = ['D', 'C', 'G']

for mat in matrices:
    mat = sp.simplify(mat)

    with open(f'comau_{names[i]}.txt', 'w') as f:
        f.write(sp.latex(mat))
    
    i += 1

