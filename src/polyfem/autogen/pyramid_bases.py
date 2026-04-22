# https://raw.githubusercontent.com/sympy/sympy/master/examples/advanced/fem.py
from sympy import *
import os
import numpy as np
import argparse
from sympy.printing import ccode

import pretty_print

x, y, z = symbols('x,y,z')

def shuffle(a,b):
    return [a[i] for i in b]

def pyramid_space(order):
    sum = 0
    basis = []
    coeff = []
    # b1, b2, b3 = Rational(x, 1 - z), Rational(y, 1 - z), 1 - z
    b1, b2, b3 = x / (1 - z), y / (1 - z), 1 - z
    for k in range(order + 1):
        for i in range(k + 1):
            for j in range(k + 1):
                aijk = Symbol("a_%d_%d_%d" % (i, j, k))
                sum += aijk * b1**i * b2**j * b3**(k)
                basis.append(b1**i * b2**j * b3**(k))
                coeff.append(aijk)
    return sum, coeff, basis

def create_point_set(order):
    h = Rational(1, order)
    set = []
    # Base
    for i in range(order + 1):
        x = i * h
        for j in range(order + 1):
            y = j * h
            set.append((x, y, 0))
    # Apex
    set.append((0, 0, 1))
    
    # Side edges
    for i in range(1, order):
        z = i * h
        for base_v in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            set.append((base_v[0] * (1 - z), base_v[1] * (1 - z), z))

    # Side faces
    for face in [[(0, 0, 1), (0, 0, 0), (1, 0, 0)], [(0, 0, 1), (1, 0, 0), (1, 1, 0)], [(0, 0, 1), (1, 1, 0), (0, 1, 0)], [(0, 0, 1), (0, 1, 0), (0, 0, 0)]]:
        f_a, f_b, f_c = face[0], face[1], face[2]
        for i in range(1, order):
            alpha = i * h
            for j in range(1, order):
                beta = j * h
                gamma = 1 - alpha - beta
                if alpha > 0 and beta > 0 and gamma > 0:
                    x = alpha * f_a[0] + beta * f_b[0] + gamma * f_c[0] # barycentric interpolation of x-coordinate
                    y = alpha * f_a[1] + beta * f_b[1] + gamma * f_c[1] # barycentric interpolation of y-coordinate
                    z = alpha * f_a[2] + beta * f_b[2] + gamma * f_c[2] # barycentric interpolation of z-coordinate
                    set.append((x, y, z))

    # Interior 
    h_i = Rational(1, order - 1)
    for k in range(1, order-1):
        z = 1 - k * h_i # 1/2 for order 3, 2/3 and 1/3 for order 4, 3/4, 1/2, 1/4 for order 5, ...
        if k == 1:
            set.append((0.5 * (1 - z), 0.5 * (1 - z), z))
        else: # k > 1
            h_k = Rational(1, k + 1)
            for i in range(1, k + 1):
                x = i * h_k * (1 - z)
                for j in range(1, k + 1):
                    y = j * h_k * (1 - z)
                    set.append((x, y, z))

    assert len(set) == (order + 1) * (order + 2) * (2 * order + 3) // 6, f"Expected {(order + 1) * (order + 2) * (order + 3) // 6} points, but got {len(set)}"
        
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

class Pyramid:
    def __init__(self, order):
        self.order = order
        self.points = []
        self.compute_basis()
    
    def nbf(self):
        return len(self.N)
    
    def compute_basis(self):
        order = self.order
        N = []
        self.points = create_point_set(order)
        sum, coeff, basis = pyramid_space(order)

        equations = []
        for p in self.points:
            ex = sum.subs(x, p[0])
            ex = ex.subs(y, p[1])
            ex = ex.subs(z, p[2])
            equations.append(ex)

        b = eye(len(equations))
        A = create_matrix(equations, coeff)
        Ainv = A.inv()
        xx = Ainv * b

        for i in range(0, len(equations)):
            Ni = sum
            for j in range(0, len(coeff)):
                Ni = Ni.subs(coeff[j], xx[j, i])
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

    dims = [3]

    orders = [0, 1, 2, 3, 4]

    bletter = "pyramid"

    cpp = f"#include \"auto_{bletter}_bases.hpp\""
    cpp = cpp + "\n#include \"auto_b_bases.hpp\""
    cpp = cpp + "\n#include \"p_n_bases.hpp\""
    cpp = cpp + "\n\n\n" \
        "namespace polyfem {\nnamespace autogen " + "{\nnamespace " + "{\n"

    hpp = "#pragma once\n\n#include <Eigen/Dense>\n#include <cassert>\n"

    hpp = hpp + "\nnamespace polyfem {\nnamespace autogen " + "{\n"

    for dim in dims:
        assert dim == 3, "Only 3D pyramid is supported"
        print(str(dim) + "D " + bletter)
        suffix = "3d"

        unique_nodes = f"void {bletter}_nodes_{suffix}" + \
            f"(const int {bletter}, Eigen::MatrixXd &val)"

        unique_fun = f"void {bletter}_basis_value_{suffix}" + \
            f"(const int {bletter}, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"
        dunique_fun = f"void {bletter}_grad_basis_value_{suffix}" + \
            f"(const int {bletter}, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"

        hpp = hpp + unique_nodes + ";\n\n"

        hpp = hpp + unique_fun + ";\n\n"
        hpp = hpp + dunique_fun + ";\n\n"

        unique_nodes = unique_nodes + f"{{\nswitch({bletter})" + "{\n"

        unique_fun = unique_fun + "{\n"
        dunique_fun = dunique_fun + "{\n"

        unique_fun = unique_fun + f"\nswitch({bletter})" + "{\n"
        dunique_fun = dunique_fun + f"\nswitch({bletter})" + "{\n"

        vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]]

        for order in orders:
            print("\t-processing " + str(order))

            if order == 0:
                def fe(): return None
                fe.nbf = lambda: 1

                fe.N = [1]

                fe.points = [[2./5., 2./5., 1./5.]]
            else:
                fe = Pyramid(order)

            current_indices = list(range(0, len(fe.points)))
            indices = []

            # vertex coordinate
            for i in range(0, 5):
                vv = vertices[i]
                for ii in current_indices:
                    norm = 0
                    for dd in range(0, dim):
                        norm = norm + (vv[dd] - fe.points[ii][dd]) ** 2

                    if norm < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # base edge 1 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][1] != 0 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][0] - (i + 1) / order) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # base edge 2 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][0] != 1 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][1] - (i + 1) / order) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break
            
            # base edge 3 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][1] != 1 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][0] - (1 - (i + 1) / order)) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # base edge 4 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][0] != 0 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][1] - (1 - (i + 1) / order)) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break
            

            # side edge 1 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][0] != 0 or fe.points[ii][1] != 0:
                        continue

                    if abs(fe.points[ii][2] - (i + 1) / order) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # side edge 2 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][0] + fe.points[ii][2] != 1 or fe.points[ii][1] != 0:
                        continue

                    if abs(fe.points[ii][0] - (1 - (i + 1) / order)) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break
            
            # side edge 3 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][0] + fe.points[ii][2] != 1 or fe.points[ii][1] + fe.points[ii][2] != 1:
                        continue
                    

                    if abs(fe.points[ii][2] - (i + 1) / order) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # side edge 4 coordinate
            for i in range(0, order - 1):
                for ii in current_indices:
                    if fe.points[ii][1] + fe.points[ii][2] != 1 or fe.points[ii][0] != 0:
                        continue

                    if abs(fe.points[ii][1] - (1 - (i + 1) / order)) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            nn = max(0, order - 2)
            npts_b = (nn + 1)**2
            npts = int(nn * (nn + 1) / 2)

            # front: y = 0  (f[0]: v0,v1,v4)
            tmp = []
            for i in range(0, npts):
                for ii in current_indices:
                    if abs(fe.points[ii][1]) > 1e-10:
                        continue
                    tmp.append(ii); current_indices.remove(ii); break
            for i in range(len(tmp)):
                indices.append(tmp[(i + 1) % len(tmp)])

            # right: x + z = 1  (f[1]: v1,v2,v4)
            tmp = []
            for i in range(0, npts):
                for ii in current_indices:
                    if abs(fe.points[ii][0]) < 1e-10 or abs(fe.points[ii][1]) < 1e-10 or abs(fe.points[ii][2]) < 1e-10:
                        continue
                    if abs((fe.points[ii][0] + fe.points[ii][2]) - 1) > 1e-10:
                        continue
                    tmp.append(ii); current_indices.remove(ii); break
            for i in range(len(tmp)):
                indices.append(tmp[(i + 1) % len(tmp)])

            # back: y + z = 1  (f[2]: v2,v3,v4)
            tmp = []
            for i in range(0, npts):
                for ii in current_indices:
                    if abs(fe.points[ii][0]) < 1e-10 or abs(fe.points[ii][1]) < 1e-10 or abs(fe.points[ii][2]) < 1e-10:
                        continue
                    if abs((fe.points[ii][1] + fe.points[ii][2]) - 1) > 1e-10:
                        continue
                    tmp.append(ii); current_indices.remove(ii); break
            for i in range(len(tmp)):
                indices.append(tmp[(i + 1) % len(tmp)])

            # left: x = 0  (f[3]: v3,v0,v4)
            tmp = []
            for i in range(0, npts):
                for ii in current_indices:
                    if abs(fe.points[ii][0]) > 1e-10:
                        continue
                    tmp.append(ii); current_indices.remove(ii); break
            for i in range(len(tmp)):
                indices.append(tmp[(i + 1) % len(tmp)])

            # bottom: z = 0  (f[4]: base quad, (p-1)^2 nodes)  ← moved to after tri faces
            for i in range(0, npts_b):
                for ii in current_indices:
                    if abs(fe.points[ii][2]) > 1e-10:
                        continue
                    indices.append(ii); current_indices.remove(ii); break

            # interior unshared indices, order does not matter
            for ii in current_indices:
                indices.append(ii)

            for i in range(0, fe.nbf()):
                print(i, indices[i], fe.points[indices[i]])

            # nodes code gen
            nodes = f"void {bletter}_{order}_nodes_{suffix}(Eigen::MatrixXd &res) {{\n res.resize(" + str(
                len(indices)) + ", " + str(dim) + "); res << \n"
            unique_nodes = unique_nodes + f"\tcase {order}: " + \
                f"{bletter}_{order}_nodes_{suffix}(val); break;\n"

            for ii in indices:
                nodes = nodes + ccode(fe.points[ii][0]) + ", " + ccode(fe.points[ii][1]) + (
                    (", " + ccode(fe.points[ii][2])) if dim == 3 else "") + ",\n"
            nodes = nodes[:-2]
            nodes = nodes + ";\n}"

            # bases code gen
            func = f"void {bletter}_{order}_basis_value_{suffix}" + \
                "(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)"
            dfunc = f"void {bletter}_{order}_basis_grad_value_{suffix}" + \
                "(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"

            unique_fun = unique_fun + \
                f"\tcase {order}: {bletter}_{order}_basis_value_{suffix}(local_index, uv, val); break;\n"
            dunique_fun = dunique_fun + \
                f"\tcase {order}: {bletter}_{order}_basis_grad_value_{suffix}(local_index, uv, val); break;\n"

            # hpp = hpp + func + ";\n"
            # hpp = hpp + dfunc + ";\n"

            base = "auto x=uv.col(0).array();\nauto y=uv.col(1).array();"
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
            cpp = cpp + dbase + "}\n\n\n"

            cpp = cpp + nodes + "\n\n\n"

        unique_nodes = unique_nodes + "\tdefault: assert(false);\n}}"

        unique_fun = unique_fun + "\tdefault: assert(false); \n}}"
        dunique_fun = dunique_fun + "\tdefault: assert(false); \n}}"
        
        cpp = cpp + "}\n\n" + unique_nodes + "\n" + unique_fun + \
            "\n\n" + dunique_fun + "\n" + "\nnamespace " + "{\n"
        hpp = hpp + "\n"

    hpp = hpp + \
        f"\nstatic const int MAX_{bletter.capitalize()}_BASES = {max(orders)};\n"

    cpp = cpp + "\n}}}\n"
    hpp = hpp + "\n}}\n"

    path = os.path.abspath(args.output)

    print("saving...")
    with open(os.path.join(path, f"auto_{bletter}_bases.cpp"), "w") as file:
        file.write(cpp)

    with open(os.path.join(path, f"auto_{bletter}_bases.hpp"), "w") as file:
        file.write(hpp)

    print("done!")


    # print("Creating point set...")
    # point_set = create_point_set(5)
    # # plot the point set
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # xs = [p[0] for p in point_set]
    # ys = [p[1] for p in point_set]
    # zs = [p[2] for p in point_set]
    # colors = [p[3] for p in point_set]
    # ax.scatter(xs, ys, zs, c=colors)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()

    # python_pyramid_basis_0 = Pyramid(0)
    # print(python_pyramid_basis_0.N, len(python_pyramid_basis_0.N))

    # python_pyramid_basis_1 = Pyramid(1)
    # print(python_pyramid_basis_1.N, len(python_pyramid_basis_1.N))

    # python_pyramid_basis_2 = Pyramid(2)
    # print(python_pyramid_basis_2.N, len(python_pyramid_basis_2.N))