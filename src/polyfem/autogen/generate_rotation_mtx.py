from sympy import symbols, trace, det, log, Pow, Rational, MatrixSymbol, Matrix, ccode, sin, cos
from sympy import *
from sympy.matrices import *
from sympy.printing import ccode
from pretty_print import C99_print_tensor, C99_print
import re

x, y, z, alpha = symbols("point(0) point(1) point(2) alpha")

cx, cy, cz = symbols("control_pt(0) control_pt(1) control_pt(2)")


def rot_mtx(angle, axis):
    R = None
    if axis == 0:
        R = Matrix([[1, 0, 0], [0, cos(angle), -sin(angle)],
                   [0, sin(angle), cos(angle)]])
    elif axis == 1:
        R = Matrix([[cos(angle), 0, sin(angle)], [0, 1, 0],
                   [-sin(angle), 0,  cos(angle)]])
    elif axis == 2:
        R = Matrix([[cos(angle), -sin(angle), 0],
                   [sin(angle), cos(angle), 0], [0, 0, 1]])
    else:
        exit()
    return R


def rot_val(params):
    p = Matrix([[x], [y], [z]])
    R_mtx = rot_mtx(params[2], 2) @ rot_mtx(params[1],
                                            1) @ rot_mtx(params[0], 0)

    p -= Matrix([[cx], [cy], [cz]])
    p = R_mtx @ p
    p += Matrix([[cx], [cy], [cz]])
    for i in range(3):
        p[i] += params[3 + i]
    return p


dim = 6
T = MatrixSymbol('param', dim, 1)
T_ = Matrix(T)

energy = rot_val(T_)
grad = energy.diff(T_)

e = []
for i in range(3):
    e.append(energy[i])
print(C99_print(e))
# exit()

grad = grad[:, 0, :, 0]
g = []
for j in range(3):
    for i in range(6):
        g.append(grad[i, j])

print(C99_print(g))
