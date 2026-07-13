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
    def __init__(self, nsd, order, bernstein):
        self.nsd = nsd
        self.bernstein = bernstein
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

        b = eye(len(equations))
        if self.bernstein:
            xx = b
        else:
            A = create_matrix(equations, coeffs)

            # if A.shape[0] > 25:
            #     A = A.evalf()

            Ainv = A.inv()
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
    parser.add_argument("--bernstein", default=False, action='store_true',
                        help="use Bernstein basis instead of Lagrange basis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dims = [2, 3]

    orders = [0, 1, 2, 3, 4]
    # orders = [4]

    bletter = "b" if args.bernstein else "p"

    cpp = f"#include \"auto_{bletter}_bases.hpp\"\n"
    if not args.bernstein:
        cpp += "#include \"p_n_bases.hpp\"\n"
    cpp += "\n"
    cpp += "namespace polyfem {\nnamespace autogen {\n"

    hpp = "#pragma once\n\n"
    hpp += "#include <polyfem/utils/CudaBoth.hpp>\n"
    hpp += "#include <polyfem/utils/Span.hpp>\n"
    hpp += "#include <cassert>\n"
    hpp += "#include <cstddef>\n"
    hpp += "#include <cmath>\n"
    if not args.bernstein:
        hpp += "\n#include \"auto_b_bases.hpp\"\n"
    hpp += "\nnamespace polyfem {\nnamespace autogen " + "{\n"

    nodes_hpp = None
    nodes_cpp = None
    if not args.bernstein:
        nodes_hpp = "#pragma once\n\n#include <Eigen/Dense>\n\n"
        nodes_hpp += "namespace polyfem {\nnamespace autogen {\n"
        nodes_cpp = "#include \"auto_p_bases_nodes.hpp\"\n"
        nodes_cpp += "#include \"p_n_bases.hpp\"\n\n"
        nodes_cpp += "namespace polyfem {\nnamespace autogen {\nnamespace {\n"

    for dim in dims:
        assert dim in (2, 3), "P simplex autogen supports only triangles and tetrahedra"
        print(str(dim) + "D " + bletter)
        suffix = "2d" if dim == 2 else "3d"

        if not args.bernstein:
            unique_nodes = f"void {bletter}_nodes_{suffix}" + \
                f"(const int {bletter}, Eigen::MatrixXd &val)"

        if args.bernstein:
            unique_fun = f"POLYFEM_BOTH void {bletter}_basis_value_{suffix}" + \
                f"(const int {bletter}, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val)"
            dunique_fun = f"POLYFEM_BOTH void {bletter}_grad_basis_value_{suffix}" + \
                f"(const int {bletter}, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z)"
        else:
            unique_fun = f"POLYFEM_BOTH void {bletter}_basis_value_{suffix}" + \
                f"(const bool bernstein, const int {bletter}, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val)"
            dunique_fun = f"POLYFEM_BOTH void {bletter}_grad_basis_value_{suffix}" + \
                f"(const bool bernstein, const int {bletter}, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z)"

        count_fun = f"POLYFEM_BOTH int {bletter}_basis_count_{suffix}(const int {bletter})"
        count_cases = f"{count_fun} {{\nswitch({bletter}) {{\n"

        if not args.bernstein:
            nodes_hpp = nodes_hpp + unique_nodes + ";\n\n"

        hpp = hpp + count_fun + ";\n\n"
        hpp = hpp + unique_fun + ";\n\n"
        hpp = hpp + dunique_fun + ";\n\n"

        if not args.bernstein:
            unique_nodes = unique_nodes + f"{{\nswitch({bletter})" + "{\n"

        unique_fun = unique_fun + "{\n"
        dunique_fun = dunique_fun + "{\n"

        if not args.bernstein:
            unique_fun = unique_fun + \
                f"if(bernstein) {{ b_basis_value_{suffix}(p, local_index, x, y, z, val); return; }}\n\n"
            dunique_fun = dunique_fun + \
                f"if(bernstein) {{ b_grad_basis_value_{suffix}(p, local_index, x, y, z, grad_x, grad_y, grad_z); return; }}\n\n"

        unique_fun = unique_fun + f"\nswitch({bletter})" + "{\n"
        dunique_fun = dunique_fun + f"\nswitch({bletter})" + "{\n"

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
                fe = Lagrange(dim, order, args.bernstein)

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
            nodes = f"void {bletter}_{order}_nodes_{suffix}(Eigen::MatrixXd &res) {{\n res.resize(" + str(
                len(indices)) + ", " + str(dim) + "); res << \n"
            if not args.bernstein:
                unique_nodes = unique_nodes + f"\tcase {order}: " + \
                    f"{bletter}_{order}_nodes_{suffix}(val); break;\n"

            for ii in indices:
                nodes = nodes + ccode(fe.points[ii][0]) + ", " + ccode(fe.points[ii][1]) + (
                    (", " + ccode(fe.points[ii][2])) if dim == 3 else "") + ",\n"
            nodes = nodes[:-2]
            nodes = nodes + ";\n}"

            # bases code gen
            # Generate two functions:
            # - "func" to eval basis value
            # - "dfunc" to eval basis gradient.
            # Both function evaluates quadrature points in batch by dispatching the "xxx_single" basis
            # kernel inside a for loop. In this script the single kernel related codegen variable
            # is denoted with scalar_ prefix.
            func = f"POLYFEM_BOTH void {bletter}_{order}_basis_value_{suffix}" + \
                "(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val)"
            dfunc = f"POLYFEM_BOTH void {bletter}_{order}_basis_grad_value_{suffix}" + \
                "(const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z)"
            scalar_func_name = f"{bletter}_{order}_basis_value_{suffix}_single"
            scalar_dfunc_name = f"{bletter}_{order}_basis_grad_value_{suffix}_single"

            unique_fun = unique_fun + \
                f"\tcase {order}: {bletter}_{order}_basis_value_{suffix}(local_index, x, y, z, val); break;\n"
            dunique_fun = dunique_fun + \
                f"\tcase {order}: {bletter}_{order}_basis_grad_value_{suffix}(local_index, x, y, z, grad_x, grad_y, grad_z); break;\n"
            count_cases = count_cases + f"\tcase {order}: return {fe.nbf()};\n"

            # hpp = hpp + func + ";\n"
            # hpp = hpp + dfunc + ";\n"

            if not args.bernstein:
                default_nodes = "p_n_nodes_3d(p, val);" if dim == 3 else "p_n_nodes_2d(p, val);"

            # Single basis kernel.
            base = ""
            dbase = ""
            # Batch basis kernel. Basically a switch dispatch + for loop.
            base_cases = "switch(local_index){\n"
            dbase_cases = "switch(local_index){\n"

            for i in range(0, fe.nbf()):
                real_index = indices[i]
                value_name = f"{scalar_func_name}_{i}"
                grad_name = f"{scalar_dfunc_name}_{i}"
                basis = simplify(fe.N[real_index])

                base = base + pretty_print.C99_print_scalar_value_function(value_name, basis, dim)
                dbase = dbase + pretty_print.C99_print_scalar_gradient_function(grad_name, basis, dim)
                base_cases = base_cases + pretty_print.C99_print_scalar_value_case(i, value_name, dim)
                dbase_cases = dbase_cases + pretty_print.C99_print_scalar_gradient_case(i, grad_name, dim)

            base_cases = base_cases + "\tdefault: assert(false);\n}"
            dbase_cases = dbase_cases + "\tdefault: assert(false);\n}"

            cpp = cpp + base + "\n\n"
            cpp = cpp + func + "{\n"
            cpp = cpp + "assert(val.size() == x.size());\n"
            if dim >= 2:
                cpp = cpp + "assert(y.size() == x.size());\n"
            if dim >= 3:
                cpp = cpp + "assert(z.size() == x.size());\n"
            cpp = cpp + base_cases + "\n}\n"

            cpp = cpp + dbase + "\n\n"
            cpp = cpp + dfunc + "{\n"
            cpp = cpp + "assert(grad_x.size() == x.size());\n"
            if dim >= 2:
                cpp = cpp + "assert(y.size() == x.size());\n"
                cpp = cpp + "assert(grad_y.size() == x.size());\n"
            if dim >= 3:
                cpp = cpp + "assert(z.size() == x.size());\n"
                cpp = cpp + "assert(grad_z.size() == x.size());\n"
            cpp = cpp + f"double gradient[{dim}];\n"
            cpp = cpp + dbase_cases + "\n}\n\n\n"

            if not args.bernstein:
                nodes_cpp = nodes_cpp + nodes + "\n\n\n"

        if args.bernstein:
            unique_nodes = ""
        else:
            unique_nodes = unique_nodes + "\tdefault: "+default_nodes+"\n}}"
            nodes_cpp = nodes_cpp + "}\n\n" + unique_nodes + "\n\nnamespace {\n"

        if args.bernstein:
            unique_fun = unique_fun + "\tdefault: assert(false); \n}}"
            dunique_fun = dunique_fun + "\tdefault: assert(false); \n}}"
        else:
            host_value = f"p_n_basis_value_{suffix}(p, local_index, x, y" + (", z" if dim == 3 else "") + ", val);"
            host_grad = f"p_n_basis_grad_value_{suffix}(p, local_index, x, y" + (", z" if dim == 3 else "") + ", grad_x, grad_y" + (", grad_z" if dim == 3 else "") + ");"
            fallback = "\tdefault:\n#ifdef __CUDA_ARCH__\n\t\tassert(false && \"generic Pn evaluation is host-only\");\n#else\n\t\t{}\n#endif\n}}}}"
            unique_fun = unique_fun + fallback.format(host_value)
            dunique_fun = dunique_fun + fallback.format(host_grad)

        count_formula = "(p + 1) * (p + 2) / 2" if dim == 2 else "(p + 1) * (p + 2) * (p + 3) / 6"
        count_cases = count_cases + f"\tdefault: return {count_formula};\n}}}}\n"

        cpp = cpp + count_cases + "\n"
        cpp = cpp + unique_fun + "\n\n" + dunique_fun + "\n\n"

    hpp = hpp + \
        f"\nstatic const int MAX_{bletter.capitalize()}_BASES = {max(orders)};\n"

    hpp = hpp + "\n}}\n"
    cpp = cpp + "\n}}\n"
    if not args.bernstein:
        nodes_cpp = nodes_cpp + "\n}}}\n"
        nodes_hpp = nodes_hpp + "}}\n"

    path = os.path.abspath(args.output)

    print("saving...")
    with open(os.path.join(path, f"auto_{bletter}_bases.cpp"), "w") as file:
        file.write(cpp)

    with open(os.path.join(path, f"auto_{bletter}_bases.hpp"), "w") as file:
        file.write(hpp)
    if not args.bernstein:
        with open(os.path.join(path, "auto_p_bases_nodes.cpp"), "w") as file:
            file.write(nodes_cpp)
        with open(os.path.join(path, "auto_p_bases_nodes.hpp"), "w") as file:
            file.write(nodes_hpp)

    print("done!")
