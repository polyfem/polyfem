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
    args = parse_args()
    dims = [2, 3]
    names = ["hooke", "saint_venant", "neo_hookean"]
    cpp = "#include \"auto_rhs.hpp\"\n\n\n"
    hpp = "#pragma once\n\n#include \"ElasticityUtils.hpp\"\n#include \"AutodiffTypes.hpp\"\n#include <Eigen/Dense>\n\n"
    cpp = cpp + "namespace poly_fem {\nnamespace autogen " + "{\n"
    hpp = hpp + "namespace poly_fem {\nnamespace autogen " + "{\n"

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    for name in names:
        for dim in dims:
            print("processing " + name + " in " + str(dim) + "D")

            if dim == 2:
                X = Matrix([x, y])
                f = Matrix([Function('f0')(x, y), Function('f1')(x, y)])
            else:
                X = Matrix([x, y, z])
                f = Matrix([Function('f0')(x, y, z), Function('f1')(x, y, z), Function('f2')(x, y, z)])

            disp_grad = f.jacobian(X)
            def_grad = (eye(dim) + disp_grad)

            if name == "hooke":
                sigma = hooke(disp_grad, def_grad)
            elif name == "saint_venant":
                sigma = saint_venant(disp_grad, def_grad)
            elif name == "neo_hookean":
                sigma = neo_hookean(disp_grad, def_grad)

            div = divergence(sigma)
            # div = simplify(div)

            c99 = pretty_print.C99_print(div)
            c99 = re.sub("f0\(x, y(, z)?\)", "pt(0)", c99)
            c99 = re.sub("f1\(x, y(, z)?\)", "pt(1)", c99)
            c99 = re.sub("f2\(x, y(, z)?\)", "pt(2)", c99)

            c99 = re.sub(", x, x\)", ".getHessian()(0,0)", c99)
            c99 = re.sub(", x, y\)", ".getHessian()(0,1)", c99)
            c99 = re.sub(", x, z\)", ".getHessian()(0,2)", c99)

            c99 = re.sub(", y, y\)", ".getHessian()(1,1)", c99)
            c99 = re.sub(", y, z\)", ".getHessian()(1,2)", c99)

            c99 = re.sub(", z, z\)", ".getHessian()(2,2)", c99)

            c99 = re.sub(", x\)", ".getGradient()(0)", c99)
            c99 = re.sub(", y\)", ".getGradient()(1)", c99)
            c99 = re.sub(", z\)", ".getGradient()(2)", c99)

            c99 = re.sub("Derivative\(*", "", c99)

            c99 = c99.replace("result_0[0]", "res(0)")
            c99 = c99.replace("result_0[1]", "res(1)")
            c99 = c99.replace("result_0[2]", "res(2)")

            # c99 = re.sub("// ", "", c99)

            signature = "void " + name + "_" + str(dim) + "d_function(const AutodiffHessianPt &pt"
            if name == "hooke" or name == "saint_venant":
                signature = signature + ", const ElasticityTensor &C"
            elif name == "neo_hookean":
                signature = signature + ", const double lambda, const double mu"

            signature = signature + ", Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res)"

            cpp = cpp + signature + " {\nres.resize(" + str(dim) + ");\n" + c99 + "\n}\n\n"
            hpp = hpp + signature + ";\n"

        cpp = cpp + "\n"
        hpp = hpp + "\n"

    cpp = cpp + "\n}}\n"
    hpp = hpp + "\n}}\n"
    path = os.path.abspath(args.output)

    print("saving...")
    with open(os.path.join(path, "auto_rhs.cpp"), "w") as file:
        file.write(cpp)

    with open(os.path.join(path, "auto_rhs.hpp"), "w") as file:
        file.write(hpp)

    print("done!")
