# https://raw.githubusercontent.com/sympy/sympy/master/examples/advanced/fem.py
from sympy import *
import os
import numpy as np
import argparse
from sympy.printing import ccode


import pretty_print

x, y, z = symbols('x,y,z')


class ReferenceSimplex:
    def __init__(self, nsd):
        self.nsd = nsd
        if nsd <= 3:
            coords = symbols('x,y,z')[:nsd]
        else:
            coords = [Symbol("x_%d" % d) for d in range(nsd)]
        self.coords = coords

    def integrate(self, f):
        coords = self.coords
        nsd = self.nsd

        limit = 1
        for p in coords:
            limit -= p

        intf = f
        for d in range(0, nsd):
            p = coords[d]
            limit += p
            intf = integrate(intf, (p, 0, limit))
        return intf


def bernstein_space(order, nsd):
    if nsd > 3:
        raise RuntimeError("Bernstein only implemented in 1D, 2D, and 3D")
    sum = 0
    basis = []
    coeff = []

    if nsd == 2:
        b1, b2, b3 = x, y, 1 - x - y
        for o1 in range(0, order + 1):
            for o2 in range(0, order + 1):
                for o3 in range(0, order + 1):
                    if o1 + o2 + o3 == order:
                        aij = Symbol("a_%d_%d_%d" % (o1, o2, o3))
                        fac = factorial(order) / (factorial(o1) *
                                                  factorial(o2) * factorial(o3))
                        sum += aij * fac * pow(b1, o1) * \
                            pow(b2, o2) * pow(b3, o3)
                        basis.append(fac * pow(b1, o1) *
                                     pow(b2, o2) * pow(b3, o3))
                        coeff.append(aij)

    if nsd == 3:
        b1, b2, b3, b4 = x, y, z, 1 - x - y - z
        for o1 in range(0, order + 1):
            for o2 in range(0, order + 1):
                for o3 in range(0, order + 1):
                    for o4 in range(0, order + 1):
                        if o1 + o2 + o3 + o4 == order:
                            aij = Symbol("a_%d_%d_%d_%d" % (o1, o2, o3, o4))
                            fac = factorial(
                                order) / (factorial(o1) * factorial(o2) * factorial(o3) * factorial(o4))
                            sum += aij * fac * \
                                pow(b1, o1) * pow(b2, o2) * \
                                pow(b3, o3) * pow(b4, o4)
                            basis.append(fac * pow(b1, o1) * pow(b2, o2) *
                                         pow(b3, o3) * pow(b4, o4))
                            coeff.append(aij)

    return sum, coeff, basis


def create_point_set(order, nsd):
    h = Rational(1, order)
    set = []

    if nsd == 2:
        for i in range(0, order + 1):
            x = i * h
            for j in range(0, order + 1):
                y = j * h
                if x + y <= 1:
                    set.append((x, y))

    if nsd == 3:
        for i in range(0, order + 1):
            x = i * h
            for j in range(0, order + 1):
                y = j * h
                for k in range(0, order + 1):
                    z = k * h
                    if x + y + z <= 1:
                        set.append((x, y, z))

    return set


def create_matrix(equations, coeffs):
    A = zeros(len(equations))
    i = 0
    j = 0
    for j in range(0, len(coeffs)):
        c = coeffs[j]
        for i in range(0, len(equations)):
            e = equations[i]
            d, _ = reduced(e, [c])
            A[i, j] = d[0]
    return A


class Lagrange:
    def __init__(self, nsd, order):
        self.nsd = nsd
        self.order = order
        self.points = []
        self.compute_basis()

    def nbf(self):
        return len(self.N)

    def compute_basis(self):
        order = self.order
        nsd = self.nsd
        N = []
        pol, coeffs, basis = bernstein_space(order, nsd)
        self.points = create_point_set(order, nsd)

        equations = []
        for p in self.points:
            ex = pol.subs(x, p[0])
            if nsd > 1:
                ex = ex.subs(y, p[1])
            if nsd > 2:
                ex = ex.subs(z, p[2])
            equations.append(ex)

        A = create_matrix(equations, coeffs)

        # if A.shape[0] > 25:
        #     A = A.evalf()

        Ainv = A.inv()

        b = eye(len(equations))

        xx = Ainv * b

        for i in range(0, len(equations)):
            Ni = pol
            for j in range(0, len(coeffs)):
                Ni = Ni.subs(coeffs[j], xx[j, i])
            N.append(Ni)

        self.N = N


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=str, help="path to the output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dims = [2, 3]

    orders = [0, 1, 2, 3, 4]
    # orders = [4]

    cpp = "#include \"auto_p_bases.hpp\"\n\n\n"
    cpp = cpp + \
        "namespace polyfem {\nnamespace autogen " + "{\nnamespace " + "{\n"

    hpp = "#pragma once\n\n#include <Eigen/Dense>\n#include \"p_n_bases.hpp\"\n#include <cassert>\n\n"
    hpp = hpp + "namespace polyfem {\nnamespace autogen " + "{\n"

    for dim in dims:
        print(str(dim) + "D")
        suffix = "_2d" if dim == 2 else "_3d"

        unique_nodes = "void p_nodes" + suffix + \
            "(const int p, Eigen::MatrixXd &val)"

        unique_fun = "void p_basis_value" + suffix + \
            "(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"
        dunique_fun = "void p_grad_basis_value" + suffix + \
            "(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"

        hpp = hpp + unique_nodes + ";\n\n"

        hpp = hpp + unique_fun + ";\n\n"
        hpp = hpp + dunique_fun + ";\n\n"

        unique_nodes = unique_nodes + "{\nswitch(p)" + "{\n"

        unique_fun = unique_fun + "{\nswitch(p)" + "{\n"
        dunique_fun = dunique_fun + "{\nswitch(p)" + "{\n"

        if dim == 2:
            vertices = [[0, 0], [1, 0], [0, 1]]
        elif dim == 3:
            vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for order in orders:
            print("\t-processing " + str(order))

            if order == 0:
                def fe(): return None
                fe.nbf = lambda: 1

                fe.N = [1]

                if dim == 2:
                    fe.points = [[1./3., 1./3.]]
                else:
                    fe.points = [[1./3., 1./3., 1./3.]]
            else:
                fe = Lagrange(dim, order)

            current_indices = list(range(0, len(fe.points)))
            indices = []

            # vertex coordinate
            for i in range(0, dim + 1):
                vv = vertices[i]
                for ii in current_indices:
                    norm = 0
                    for dd in range(0, dim):
                        norm = norm + (vv[dd] - fe.points[ii][dd]) ** 2

                    if norm < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # edge 1 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][1] != 0 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][0] - (i + 1) / order) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # edge 2 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][0] + fe.points[ii][1] != 1 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][1] - (i + 1) / order) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # edge 3 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][0] != 0 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][1] - (1 - (i + 1) / order)) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            if dim == 3:
                # edge 4 coordinate
                for i in range(0, order - 1):
                    for ii in current_indices:
                        if fe.points[ii][0] != 0 or fe.points[ii][1] != 0:
                            continue

                        if abs(fe.points[ii][2] - (i + 1) / order) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 5 coordinate
                for i in range(0, order - 1):
                    for ii in current_indices:
                        if fe.points[ii][0] + fe.points[ii][2] != 1 or fe.points[ii][1] != 0:
                            continue

                        if abs(fe.points[ii][0] - (1 - (i + 1) / order)) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 6 coordinate
                for i in range(0, order - 1):
                    for ii in current_indices:
                        if fe.points[ii][1] + fe.points[ii][2] != 1 or fe.points[ii][0] != 0:
                            continue

                        if abs(fe.points[ii][1] - (1 - (i + 1) / order)) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

            if dim == 3:
                nn = max(0, order - 2)
                npts = int(nn * (nn + 1) / 2)

                # bottom: z = 0
                for i in range(0, npts):
                    for ii in current_indices:
                        if abs(fe.points[ii][2]) > 1e-10:
                            continue

                        indices.append(ii)
                        current_indices.remove(ii)
                        break

                # front: y = 0
                for i in range(0, npts):
                    for ii in current_indices:
                        if abs(fe.points[ii][1]) > 1e-10:
                            continue

                        indices.append(ii)
                        current_indices.remove(ii)
                        break

                # diagonal: none equal to zero and sum 1
                tmp = []
                for i in range(0, npts):
                    for ii in current_indices:
                        if (abs(fe.points[ii][0]) < 1e-10) | (abs(fe.points[ii][1]) < 1e-10) | (abs(fe.points[ii][2]) < 1e-10):
                            continue

                        if abs((fe.points[ii][0] + fe.points[ii][1] + fe.points[ii][2]) - 1) > 1e-10:
                            continue

                        tmp.append(ii)
                        current_indices.remove(ii)
                        break
                for i in range(0, len(tmp)):
                    indices.append(tmp[(i + 2) % len(tmp)])

                # side: x = 0
                tmp = []
                for i in range(0, npts):
                    for ii in current_indices:
                        if abs(fe.points[ii][0]) > 1e-10:
                            continue

                        tmp.append(ii)
                        current_indices.remove(ii)
                        break
                tmp.sort(reverse=True)
                indices.extend(tmp)

            # either face or volume indices, order do not matter
            for ii in current_indices:
                indices.append(ii)

            # nodes code gen
            nodes = "void p_" + str(order) + "_nodes" + suffix + "(Eigen::MatrixXd &res) {\n res.resize(" + str(
                len(indices)) + ", " + str(dim) + "); res << \n"
            unique_nodes = unique_nodes + "\tcase " + \
                str(order) + ": " + "p_" + str(order) + \
                "_nodes" + suffix + "(val); break;\n"

            for ii in indices:
                nodes = nodes + ccode(fe.points[ii][0]) + ", " + ccode(fe.points[ii][1]) + (
                    (", " + ccode(fe.points[ii][2])) if dim == 3 else "") + ",\n"
            nodes = nodes[:-2]
            nodes = nodes + ";\n}"

            # bases code gen
            func = "void p_" + str(order) + "_basis_value" + suffix + \
                "(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)"
            dfunc = "void p_" + str(order) + "_basis_grad_value" + suffix + \
                "(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"

            unique_fun = unique_fun + "\tcase " + str(order) + ": " + "p_" + str(
                order) + "_basis_value" + suffix + "(local_index, uv, val); break;\n"
            dunique_fun = dunique_fun + "\tcase " + str(order) + ": " + "p_" + str(
                order) + "_basis_grad_value" + suffix + "(local_index, uv, val); break;\n"

            # hpp = hpp + func + ";\n"
            # hpp = hpp + dfunc + ";\n"

            default_base = "p_n_basis_value_3d(p, local_index, uv, val);" if dim == 3 else "p_n_basis_value_2d(p, local_index, uv, val);"
            default_dbase = "p_n_basis_grad_value_3d(p, local_index, uv, val);" if dim == 3 else "p_n_basis_grad_value_2d(p, local_index, uv, val);"
            default_nodes = "p_n_nodes_3d(p, val);" if dim == 3 else "p_n_nodes_2d(p, val);"

            base = "auto x=uv.col(0).array();\nauto y=uv.col(1).array();"
            if dim == 3:
                base = base + "\nauto z=uv.col(2).array();"
            base = base + "\n\n"
            dbase = base

            if order == 0:
                base = base + "result_0.resize(x.size(),1);\n"

            base = base + "switch(local_index){\n"
            dbase = dbase + \
                "val.resize(uv.rows(), uv.cols());\n Eigen::ArrayXd result_0(uv.rows());\n" + \
                "switch(local_index){\n"

            for i in range(0, fe.nbf()):
                real_index = indices[i]
                # real_index = i

                base = base + "\tcase " + str(i) + ": {" + pretty_print.C99_print(
                    simplify(fe.N[real_index])).replace(" = 1;", ".setOnes();") + "} break;\n"
                dbase = dbase + "\tcase " + str(i) + ": {" + \
                    "{" + pretty_print.C99_print(simplify(diff(fe.N[real_index], x))).replace(" = 0;", ".setZero();").replace(" = 1;", ".setOnes();").replace(" = -1;", ".setConstant(-1);") + "val.col(0) = result_0; }" \
                    "{" + pretty_print.C99_print(simplify(diff(fe.N[real_index], y))).replace(" = 0;", ".setZero();").replace(
                        " = 1;", ".setOnes();").replace(" = -1;", ".setConstant(-1);") + "val.col(1) = result_0; }"

                if dim == 3:
                    dbase = dbase + "{" + pretty_print.C99_print(simplify(diff(fe.N[real_index], z))).replace(" = 0;", ".setZero();").replace(
                        " = 1;", ".setOnes();").replace(" = -1;", ".setConstant(-1);") + "val.col(2) = result_0; }"

                dbase = dbase + "} break;\n"

            base = base + "\tdefault: assert(false);\n}"
            dbase = dbase + "\tdefault: assert(false);\n}"

            cpp = cpp + func + "{\n\n"
            cpp = cpp + base + "}\n"

            cpp = cpp + dfunc + "{\n\n"
            cpp = cpp + dbase + "}\n\n\n" + nodes + "\n\n\n"

        unique_nodes = unique_nodes + "\tdefault: "+default_nodes+"\n}}"

        unique_fun = unique_fun + "\tdefault: "+default_base+"\n}}"
        dunique_fun = dunique_fun + "\tdefault: "+default_dbase+"\n}}"

        cpp = cpp + "}\n\n" + unique_nodes + "\n" + unique_fun + \
            "\n\n" + dunique_fun + "\n" + "\nnamespace " + "{\n"
        hpp = hpp + "\n"

    hpp = hpp + "\nstatic const int MAX_P_BASES = " + str(max(orders)) + ";\n"

    cpp = cpp + "\n}}}\n"
    hpp = hpp + "\n}}\n"

    path = os.path.abspath(args.output)

    print("saving...")
    with open(os.path.join(path, "auto_p_bases.cpp"), "w") as file:
        file.write(cpp)

    with open(os.path.join(path, "auto_p_bases.hpp"), "w") as file:
        file.write(hpp)

    print("done!")
