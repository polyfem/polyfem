from sympy import *
from sympy.matrices import *
import os
import re
import argparse

# local
import pretty_print


def sqr(a):
    return a * a


def trunc_acos(x):
    tmp = Piecewise((0.0, x >= 1.0), (pi, x <= -1.0), (acos(x), True))
    return tmp.subs(x, x)


def eigs_2d(mat):
    a = mat[0, 0] + mat[1, 1]
    delta = (mat[0, 0] - mat[1, 1])**2 + 4 * mat[0, 1]**2

    tmp1 = Piecewise(
        (a / 2, delta < 1e-10),
        ((a - sqrt(delta)) / 2.0, True)
    )

    tmp2 = Piecewise(
        (a / 2, delta < 1e-10),
        ((a + sqrt(delta)) / 2.0, True)
    )

    return tmp1.subs(delta, delta), tmp2.subs(delta, delta)


def eigs_3d(mat):
    b = mat[0] + mat[4] + mat[8]
    t = sqr(mat[1]) + sqr(mat[2]) + sqr(mat[5])
    p = 0.5 * (sqr(mat[0] - mat[4]) + sqr(mat[0] - mat[8]) + sqr(mat[4] - mat[8]))
    p += 3.0 * t
    q = 18.0 * (mat[0] * mat[4] * mat[8] + 3.0 * mat[1] * mat[2] * mat[5])
    q += 2.0 * (mat[0] * sqr(mat[0]) + mat[4] * sqr(mat[4]) + mat[8] * sqr(mat[8]))
    q += 9.0 * b * t
    q -= 3.0 * (mat[0] + mat[4]) * (mat[0] + mat[8]) * (mat[4] + mat[8])
    q -= 27.0 * (mat[0] * sqr(mat[5]) + mat[4] * sqr(mat[2]) + mat[8] * sqr(mat[1]))

    delta = trunc_acos(0.5 * q / sqrt(p * sqr(p)))
    p = 2.0 * sqrt(p)

    tmp1 = Piecewise(
        (b / 3.0, p < 1e-10),
        ((b + p * cos(delta / 3.0)) / 3.0, True)
    )

    tmp2 = Piecewise(
        (b / 3.0, p < 1e-10),
        ((b + p * cos((delta + 2.0 * pi) / 3.0)) / 3.0, True)
    )

    tmp3 = Piecewise(
        (b / 3.0, p < 1e-10),
        ((b + p * cos((delta - 2.0 * pi) / 3.0)) / 3.0, True)
    )

    return tmp1.subs(p, p), tmp2.subs(p, p), tmp3.subs(p, p)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=str, help="path to the output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dims = [2, 3]

    cpp = "#include <polyfem/autogen/auto_eigs.hpp>\n\n\n"
    hpp = "#pragma once\n\n#include <Eigen/Dense>\n\n"
    cpp = cpp + "namespace polyfem {\nnamespace autogen " + "{\n"
    hpp = hpp + "namespace polyfem {\nnamespace autogen " + "{\n"

    hpp = hpp + "template<typename T>\nT int_pow(T val, int exp) { T res = exp <=0 ? T(0.): val; for(int i = 1; i < exp; ++i) res = res*val; return res; }\n\n"

    lambdaa = Symbol('lambda', real=True)

    for dim in dims:
        print("processing " + str(dim))

        M = zeros(dim, dim)

        for i in range(0, dim):
            for j in range(0, dim):
                if i <= j:
                    M[i, j] = Symbol('m[' + str(i) + ',' + str(j) + ']', real=True)
                else:
                    M[i, j] = Symbol('m[' + str(j) + ',' + str(i) + ']', real=True)

        if dim == 2:
            lambdas = eigs_2d(M)
        else:
            lambdas = eigs_3d(M)

        # lambdas = simplify(lambdas)

        c99 = pretty_print.C99_print(lambdas)

        c99 = re.sub(r"m\[(\d{1}),(\d{1})\]", r'm(\1,\2)', c99)
        c99 = re.sub(r"result_(\d{1})", r'res(\1)', c99)
        c99 = c99.replace("0.0", "T(0)")
        c99 = c99.replace("   M_PI", "   T(M_PI)")

        signature = "template<typename T>\nvoid eigs_" + str(dim) + "d(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &m, "
        signature += "Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> &res)"

        hpp = hpp + signature + " {\nres.resize(" + str(dim) + ");\n" + c99 + "\n}\n\n"

    cpp = cpp + "\n"
    hpp = hpp + "\n"

    cpp = cpp + "\n}}\n"
    hpp = hpp + "\n}}\n"

    path = os.path.abspath(args.output)

    print("saving...")
    with open(os.path.join(path, "auto_eigs.cpp"), "w") as file:
        file.write(cpp)

    with open(os.path.join(path, "auto_eigs.hpp"), "w") as file:
        file.write(hpp)

    print("done!")
