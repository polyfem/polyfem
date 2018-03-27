from sympy import *
from sympy.matrices import *
import re
import os
import argparse

# local
import pretty_print


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=str, help="path to the output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dims = [2, 3]

    cpp = "#include \"auto_eigs.hpp\"\n\n\n"
    hpp = "#pragma once\n\n#include <complex>\n#include <Eigen/Dense>\n\n"
    cpp = cpp + "namespace poly_fem {\nnamespace autogen " + "{\n"
    hpp = hpp + "namespace poly_fem {\nnamespace autogen " + "{\n"

    hpp = hpp + "template<typename T>\nT int_pow(T val, int exp) { T res = exp <=0 ? T(0.): val; for(int i = 1; i < exp; ++i) res = res*val; return res; }\n\n"

    lambdaa = Symbol('lambda')

    for dim in dims:
        print("processing " + str(dim))

        M = zeros(dim, dim)

        for i in range(0, dim):
            for j in range(0, dim):
                if i <= j:
                    M[i, j] = Symbol('m[' + str(i) + ',' + str(j) + ']')
                else:
                    M[i, j] = Symbol('m[' + str(j) + ',' + str(i) + ']')

        # M = simplify(M * Transpose(M))
        # print(M)

        # if dim == 2:
        #     lambdas = roots(det(lambdaa * eye(dim) - M), lambdaa)
        # else:
        #     lambdas = roots(det(lambdaa * eye(dim) - M), lambdaa)

        # for i in range(0, dim):
        #     lambdas[i] = real(lambdas[i])

        lambdas = M.eigenvals()
        lambdas = simplify(lambdas)
        # print(lambdas)

        c99 = pretty_print.C99_print(lambdas)
        c99 = re.sub("m\[(\d{1}),(\d{1})\]", r'm(\1,\2)', c99)
        c99 = re.sub("result_(\d{1})", r'res(\1)', c99)
        c99 = re.sub("I", "1i", c99)
        c99 = re.sub("L", "", c99)
        c99 = c99.replace(" 1;", " T(1.);")
        c99 = c99.replace(" 2*", " T(2.)*")
        c99 = c99.replace(" 3*", " T(3.)*")
        c99 = c99.replace(" 4*", " T(4.)*")
        c99 = c99.replace(" 27*", " T(27.)*")
        c99 = c99.replace("-27*", "T(-27.)*")
        c99 = c99.replace(" 9*", " T(9.)*")
        c99 = c99.replace(" 54*", " T(54.)*")
        c99 = c99.replace("-4*", "T(-4.)*")
        c99 = c99.replace("27.0/2.0", "T(27.0/2.0)")
        c99 = c99.replace("9.0/2.0", "T(9.0/2.0)")
        c99 = c99.replace("1.0/2.0", "T(1.0/2.0)")
        c99 = c99.replace("1.0/3.0", "T(1.0/3.0)")

        c99 = c99.replace("pow", "int_pow")

        if dim == 3:
            c99 = re.sub("cbrt\((.*)\)", r"pow(\1,1./3.)", c99)

            c99 = re.sub("auto", "std::complex<T>", c99)
            c99 = re.sub("res\((\d)\) = (.*);", r'res(\1) = (\2).real();', c99)

        signature = "template<typename T>\nvoid eigs_" + str(dim) + "d(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &m, "
        signature += "Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> &res)"

        hpp = hpp + signature + " {\nres.resize(" + str(dim) + ");\nusing namespace std::complex_literals;\n" + c99 + "\n}\n\n"
        # hpp = hpp + signature + ";\n"

    cpp = cpp + "\n"
    hpp = hpp + "\n"

    cpp = cpp + "\n}}\n"
    hpp = hpp + "\n}}\n"

    # print(cpp)
    path = os.path.abspath(args.output)

    print("saving...")
    with open(os.path.join(path, "auto_eigs.cpp"), "w") as file:
        file.write(cpp)

    with open(os.path.join(path, "auto_eigs.hpp"), "w") as file:
        file.write(hpp)

    print("done!")
