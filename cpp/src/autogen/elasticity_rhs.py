from sympy import *
from sympy.matrices import *
import re
import os
import argparse

# local
import pretty_print


def sigma_fun(j, ee, C, dim):
    res = 0

    for k in range(dim):
        res += C(j, k) * ee[k]
    return res


# sigma = (C:strain)
def hooke(disp_grad, def_grad):
    l_dim = def_grad.rows

    C = Function('C')

    eps = 1 / 2 * (disp_grad + Transpose(disp_grad))

    if l_dim == 2:
        ee = [eps[0, 0], eps[1, 1], 2 * eps[0, 1]]

        sigma = Matrix([
            [sigma_fun(0, ee, C, 3), sigma_fun(2, ee, C, 3)],
            [sigma_fun(2, ee, C, 3), sigma_fun(1, ee, C, 3)]
        ])
    else:
        ee  = [eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]]

        sigma = Matrix([
            [sigma_fun(0, ee, C, 6), sigma_fun(5, ee, C, 6), sigma_fun(4, ee, C, 6)],
            [sigma_fun(5, ee, C, 6), sigma_fun(1, ee, C, 6), sigma_fun(3, ee, C, 6)],
            [sigma_fun(4, ee, C, 6), sigma_fun(3, ee, C, 6), sigma_fun(2, ee, C, 6)]
        ])

    return sigma


# sigma = def_grad*(C:strain)
def saint_venant(disp_grad, def_grad):
    l_dim = def_grad.rows

    C = Function('C')

    eps = 1 / 2 * (Transpose(disp_grad) * disp_grad + disp_grad + Transpose(disp_grad))

    if l_dim == 2:
        ee = [eps[0, 0], eps[1, 1], 2 * eps[0, 1]]

        sigma = Matrix([
            [sigma_fun(0, ee, C, 3), sigma_fun(2, ee, C, 3)],
            [sigma_fun(2, ee, C, 3), sigma_fun(1, ee, C, 3)]
        ])
    else:
        ee  = [eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]]

        sigma = Matrix([
            [sigma_fun(0, ee, C, 6), sigma_fun(5, ee, C, 6), sigma_fun(4, ee, C, 6)],
            [sigma_fun(5, ee, C, 6), sigma_fun(1, ee, C, 6), sigma_fun(3, ee, C, 6)],
            [sigma_fun(4, ee, C, 6), sigma_fun(3, ee, C, 6), sigma_fun(2, ee, C, 6)]
        ])

    sigma = def_grad * sigma

    return sigma


# sigma = mu * (def_grad - def_grad^{-T}) + lambda ln (det def_grad) def_grad^{-T}
def neo_hookean(disp_grad, def_grad):
    mu = Symbol('mu')
    llambda = Symbol('lambda')

    FmT = Inverse(Transpose(def_grad))

    return mu * (def_grad - FmT) + llambda * log(Determinant(def_grad)) * FmT


def divergence(sigma):
    if dim == 2:
        div = Matrix([
            sigma[0, 0].diff(x) + sigma[0, 1].diff(y),
            sigma[1, 0].diff(x) + sigma[1, 1]. diff(y)
        ])
    else:
        div = Matrix([
            sigma[0, 0].diff(x) + sigma[0, 1].diff(y) + sigma[0, 2].diff(z),
            sigma[1, 0].diff(x) + sigma[1, 1].diff(y) + sigma[1, 2].diff(z),
            sigma[2, 0].diff(x) + sigma[2, 1].diff(y) + sigma[2, 2].diff(z)
        ])

    return div


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=str, help="path to the output folder")
    return parser.parse_args()


if __name__ == "__main__":
    

    print("done!")
