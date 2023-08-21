# https://raw.githubusercontent.com/sympy/sympy/master/examples/advanced/fem.py
from sympy import *
import os
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


def create_point_set(order, nsd):
    h = Rational(1, order)
    set = []

    if nsd == 2:
        for i in range(0, order + 1):
            x = i * h
            for j in range(0, order + 1):
                y = j * h
                set.append((x, y))

    if nsd == 3:
        for i in range(0, order + 1):
            x = i * h
            for j in range(0, order + 1):
                y = j * h
                for k in range(0, order + 1):
                    z = k * h
                    set.append((x, y, z))

    return set


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
        self.points = create_point_set(order, nsd)

        if nsd == 2:
            Ntmpx = []
            Ntmpy = []

            for j in range(order + 1):
                vx = 1
                vy = 1
                xj = 1./(order)*j
                for m in range(order+1):
                    if m == j:
                        continue
                    xm = 1./(order)*m
                    vx *= (x - xm)/(xj - xm)
                    vy *= (y - xm)/(xj - xm)

                Ntmpx.append(vx)
                Ntmpy.append(vy)

            for i in range(order + 1):
                for j in range(order + 1):
                    N.append(Ntmpx[i]*Ntmpy[j])
        elif nsd == 3:
            Ntmpx = []
            Ntmpy = []
            Ntmpz = []

            for j in range(order + 1):
                vx = 1
                vy = 1
                vz = 1
                xj = 1./(order)*j
                for m in range(order+1):
                    if m == j:
                        continue
                    xm = 1./(order)*m
                    vx *= (x - xm)/(xj - xm)
                    vy *= (y - xm)/(xj - xm)
                    vz *= (z - xm)/(xj - xm)

                Ntmpx.append(vx)
                Ntmpy.append(vy)
                Ntmpz.append(vz)

            for i in range(order + 1):
                for j in range(order + 1):
                    for l in range(order + 1):
                        N.append(Ntmpx[i]*Ntmpy[j]*Ntmpz[l])

        self.N = N


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=str, help="path to the output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = os.path.abspath(args.output)

    dims = [2, 3]
    orders = [0, 1, 2, 3, -2]

    for dim in dims:
        namev = f"auto_q_bases_{dim}d_val"
        namen = f"auto_q_bases_{dim}d_nodes"
        nameg = f"auto_q_bases_{dim}d_grad"

        cppv = f"#include \"{namev}.hpp\"\n\n\n"
        cppv = cppv + "namespace polyfem {\nnamespace autogen " + "{\nnamespace " + "{\n"

        cppn = f"#include \"{namen}.hpp\"\n\n\n"
        cppn = cppn + "namespace polyfem {\nnamespace autogen " + "{\nnamespace " + "{\n"

        cppg = f"#include \"{nameg}.hpp\"\n\n\n"
        cppg = cppg + "namespace polyfem {\nnamespace autogen " + "{\nnamespace " + "{\n"
        if dim==3:
            cppg="#include <Eigen/Dense>\n namespace polyfem {\nnamespace autogen {"

        eextern=""

        hppv = "#pragma once\n\n#include <Eigen/Dense>\n\n"
        hppv = hppv + "namespace polyfem {\nnamespace autogen " + "{\n"

        hppn = "#pragma once\n\n#include <Eigen/Dense>\n\n"
        hppn = hppn + "namespace polyfem {\nnamespace autogen " + "{\n"

        hppg = "#pragma once\n\n#include <Eigen/Dense>\n\n"
        hppg = hppg + "namespace polyfem {\nnamespace autogen " + "{\n"

        print(str(dim) + "D")
        suffix = "_2d" if dim == 2 else "_3d"

        unique_nodes = "void q_nodes" + suffix + "(const int q, Eigen::MatrixXd &val)"

        unique_fun = "void q_basis_value" + suffix + "(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"
        dunique_fun = "void q_grad_basis_value" + suffix + "(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"

        hppn = hppn + unique_nodes + ";\n\n"

        hppv = hppv + unique_fun + ";\n\n"
        hppg = hppg + dunique_fun + ";\n\n"

        unique_nodes = unique_nodes + "{\nswitch(q)" + "{\n"

        unique_fun = unique_fun + "{\nswitch(q)" + "{\n"
        dunique_fun = dunique_fun + "{\nswitch(q)" + "{\n"

        if dim == 2:
            vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]
        elif dim == 3:
            vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]

        for order in orders:
            print("\t-processing " + str(order))

            if order == 0:
                fe = lambda: None
                fe.nbf = lambda: 1

                fe.N = [1]

                if dim == 2:
                    fe.points = [[1./2., 1./2.]]
                else:
                    fe.points = [[1./2., 1./2., 1./2.]]
            elif order == -2:
                fe = lambda: None
                if dim == 2:
                    fe.points = []
                    fe.N = []
                    fe.nbf = lambda: 8

                    for xi_a in [-1, 1]:
                        for eta_a in [-1, 1]:
                            tmp = (1/4*(2*x*xi_a-xi_a+1))*(2*eta_a*y-eta_a+1)*(2*eta_a*y+2*x*xi_a-eta_a-xi_a-1)
                            fe.N.append(tmp)
                            fe.points.append([(xi_a+1)/2,(eta_a+1)/2])

                    for eta_a in [-1, 1]:
                        tmp = -2*x*(x-1)*(2*eta_a*y-eta_a+1)
                        fe.N.append(tmp)
                        fe.points.append([1/2,(eta_a+1)/2])

                    for xi_a in [-1, 1]:
                        tmp = -2*y*(y-1)*(2*x*xi_a-xi_a+1)
                        fe.N.append(tmp)
                        fe.points.append([(xi_a+1)/2, 1/2])

                    assert(len(fe.points) == 8)
                    assert(len(fe.N) == 8)

                elif dim == 3:
                    fe.points = []
                    fe.N = []
                    fe.nbf = lambda: 20

                    for xi_a in [-1, 1]:
                        for eta_a in [-1, 1]:
                            for zeta_a in [-1, 1]:
                                tmp = (1/8*(2*x*xi_a-xi_a+1))*(2*eta_a*y-eta_a+1)*(2*z*zeta_a-zeta_a+1)*(2*eta_a*y+2*x*xi_a+2*z*zeta_a-eta_a-xi_a-zeta_a-2)
                                fe.N.append(tmp)
                                fe.points.append([(xi_a+1)/2,(eta_a+1)/2,(zeta_a+1)/2])

                    for eta_a in [-1, 1]:
                        for zeta_a in [-1, 1]:
                            tmp = -x*(x-1)*(2*eta_a*y-eta_a+1)*(2*z*zeta_a-zeta_a+1)
                            fe.N.append(tmp)
                            fe.points.append([1/2,(eta_a+1)/2,(zeta_a+1)/2])

                    for xi_a in [-1, 1]:
                        for zeta_a in [-1, 1]:
                            tmp = -y*(y-1)*(2*x*xi_a-xi_a+1)*(2*z*zeta_a-zeta_a+1)
                            fe.N.append(tmp)
                            fe.points.append([(xi_a+1)/2,1/2, (zeta_a+1)/2])

                    for xi_a in [-1, 1]:
                        for eta_a in [-1, 1]:
                            tmp = -z*(z-1)*(2*x*xi_a-xi_a+1)*(2*eta_a*y-eta_a+1)
                            fe.N.append(tmp)
                            fe.points.append([(xi_a+1)/2, (eta_a+1)/2, 1/2])

                    assert(len(fe.points) == 20)
                    assert(len(fe.N) == 20)
                else:
                    assert(False)

            else:
                fe = Lagrange(dim, order)


            current_indices = list(range(0, len(fe.points)))
            indices = []

            # vertex coordinate
            for i in range(0, 4*(dim-1)):
                vv = vertices[i]
                for ii in current_indices:
                    norm = 0
                    for dd in range(0, dim):
                        norm = norm + (vv[dd] - fe.points[ii][dd]) ** 2

                    if norm < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break



            # edge 0 coordinate
            for i in range(0, abs(order) - 1):
                for ii in current_indices:
                    if fe.points[ii][1] != 0 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][0] - (i + 1) / abs(order)) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # edge 1 coordinate
            for i in range(0, abs(order) - 1):
                for ii in current_indices:
                    if fe.points[ii][0] != 1 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][1] - (i + 1) / abs(order)) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # edge 2 coordinate
            for i in range(0, abs(order) - 1):
                for ii in current_indices:
                    if fe.points[ii][1] != 1 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][0] - (1 - (i + 1) / abs(order))) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            # edge 3 coordinate
            for i in range(0, abs(order) - 1):
                for ii in current_indices:
                    if fe.points[ii][0] != 0 or (dim == 3 and fe.points[ii][2] != 0):
                        continue

                    if abs(fe.points[ii][1] - (1 - (i + 1) / abs(order))) < 1e-10:
                        indices.append(ii)
                        current_indices.remove(ii)
                        break

            if dim == 3:
                # edge 4 coordinate
                for i in range(0, abs(order) - 1):
                    for ii in current_indices:
                        if fe.points[ii][0] != 0 or fe.points[ii][1] != 0:
                            continue

                        if abs(fe.points[ii][2] - (i + 1) / abs(order)) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 5 coordinate
                for i in range(0, abs(order) - 1):
                    for ii in current_indices:
                        if fe.points[ii][0] != 1 or fe.points[ii][1] != 0:
                            continue

                        if abs(fe.points[ii][2] - (1 - (i + 1) / abs(order))) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 6 coordinate
                for i in range(0, abs(order) - 1):
                    for ii in current_indices:
                        if fe.points[ii][0] != 1 or fe.points[ii][1] != 1:
                            continue

                        if abs(fe.points[ii][2] - (1 - (i + 1) / abs(order))) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 7 coordinate
                for i in range(0, abs(order) - 1):
                    for ii in current_indices:
                        if fe.points[ii][0] != 0 or fe.points[ii][1] != 1:
                            continue

                        if abs(fe.points[ii][2] - (1 - (i + 1) / abs(order))) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 8 coordinate
                for i in range(0, abs(order) - 1):
                    for ii in current_indices:
                        if fe.points[ii][1] != 0 or fe.points[ii][2] != 1:
                            continue

                        if abs(fe.points[ii][0] - (i + 1) / abs(order)) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 9 coordinate
                for i in range(0, abs(order) - 1):
                    for ii in current_indices:
                        if fe.points[ii][0] != 1 or fe.points[ii][2] != 1:
                            continue

                        if abs(fe.points[ii][1] - (i + 1) / abs(order)) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 10 coordinate
                for i in range(0, abs(order) - 1):
                    for ii in current_indices:
                        if fe.points[ii][1] != 1 or fe.points[ii][2] != 1:
                            continue

                        if abs(fe.points[ii][0] - (1 - (i + 1) / abs(order))) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

                # edge 11 coordinate
                for i in range(0, abs(order) - 1):
                    for ii in current_indices:
                        if fe.points[ii][0] != 0 or fe.points[ii][2] != 1:
                            continue

                        if abs(fe.points[ii][1] - (1 - (i + 1) / abs(order))) < 1e-10:
                            indices.append(ii)
                            current_indices.remove(ii)
                            break

            if dim == 3:
                nn = max(0, abs(order) - 1)
                npts = int(nn * nn)

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

                # side: x = 1
                tmp = []
                for i in range(0, npts):
                    for ii in current_indices:
                        if abs(fe.points[ii][0] - 1) > 1e-10:
                            continue

                        tmp.append(ii)
                        current_indices.remove(ii)
                        break
                indices.extend(tmp)

                # front: y = 0
                for i in range(0, npts):
                    for ii in current_indices:
                        if abs(fe.points[ii][1]) > 1e-10:
                            continue

                        indices.append(ii)
                        current_indices.remove(ii)
                        break

                # back: y = 1
                for i in range(0, npts):
                    for ii in current_indices:
                        if abs(fe.points[ii][1]-1) > 1e-10:
                            continue

                        indices.append(ii)
                        current_indices.remove(ii)
                        break

                # bottom: z = 0
                for i in range(0, npts):
                    for ii in current_indices:
                        if abs(fe.points[ii][2]) > 1e-10:
                            continue

                        indices.append(ii)
                        current_indices.remove(ii)
                        break

                # top: z = 1
                for i in range(0, npts):
                    for ii in current_indices:
                        if abs(fe.points[ii][2]-1) > 1e-10:
                            continue

                        indices.append(ii)
                        current_indices.remove(ii)
                        break


            # either face or volume indices, order do not matter
            for ii in current_indices:
                indices.append(ii)

            orderN = str(order) if order >= 0 else "m"+str(-order)
            # nodes code gen
            nodes = "void q_" + orderN + "_nodes" + suffix + "(Eigen::MatrixXd &res) {\n res.resize(" + str(len(indices)) + ", " + str(dim) + "); res << \n"
            unique_nodes = unique_nodes + "\tcase " + str(order) + ": " + "q_" + orderN + "_nodes" + suffix + "(val); break;\n"

            eextern = eextern + f"extern \"C++\" void q_{orderN}_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);\n"

            for ii in indices:
                nodes = nodes + ccode(fe.points[ii][0]) + ", " + ccode(fe.points[ii][1]) + ((", " + ccode(fe.points[ii][2])) if dim == 3 else "") + ",\n"
            nodes = nodes[:-2]
            nodes = nodes + ";\n}"

            # bases code gen
            func = "void q_" + orderN + "_basis_value" + suffix + "(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)"
            dfunc = "void q_" + orderN + "_basis_grad_value" + suffix + "(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)"

            unique_fun = unique_fun + "\tcase " + str(order) + ": " + "q_" + orderN + "_basis_value" + suffix + "(local_index, uv, val); break;\n"
            dunique_fun = dunique_fun + "\tcase " + str(order) + ": " + "q_" + orderN + "_basis_grad_value" + suffix + "(local_index, uv, val); break;\n"

            # hpp = hpp + func + ";\n"
            # hpp = hpp + dfunc + ";\n"

            base = "auto x=uv.col(0).array();\nauto y=uv.col(1).array();"
            if dim == 3:
                base = base + "\nauto z=uv.col(2).array();"
            base = base + "\n\n"
            dbase = base

            if order == 0:
                base = base + "result_0.resize(x.size(),1);\n"

            base = base + "switch(local_index){\n"
            dbase = dbase + "val.resize(uv.rows(), uv.cols());\n Eigen::ArrayXd result_0(uv.rows());\n" + "switch(local_index){\n"

            for i in range(0, fe.nbf()):
                real_index = indices[i]
                # real_index = i

                if dim == 3:
                    base = base + "\tcase " + str(i) + ": {" + pretty_print.C99_print(simplify(fe.N[real_index])).replace(" = 1;", ".setOnes();") + "} break;\n"
                    dbase = dbase + "\tcase " + str(i) + ": {" + \
                        "{" + pretty_print.C99_print(simplify(diff(fe.N[real_index], x))).replace(" = 0;", ".setZero();").replace(" = 1;", ".setOnes();").replace(" = -1;", ".setConstant(-1);") + "val.col(0) = result_0; }" \
                        "{" + pretty_print.C99_print(simplify(diff(fe.N[real_index], y))).replace(" = 0;", ".setZero();").replace(" = 1;", ".setOnes();").replace(" = -1;", ".setConstant(-1);") + "val.col(1) = result_0; }"
                    dbase = dbase + "{" + pretty_print.C99_print(simplify(diff(fe.N[real_index], z))).replace(" = 0;", ".setZero();").replace(" = 1;", ".setOnes();").replace(" = -1;", ".setConstant(-1);") + "val.col(2) = result_0; }"
                else:
                    base = base + "\tcase " + str(i) + ": {" + pretty_print.C99_print(simplify(fe.N[real_index])).replace(" = 1;", ".setOnes();") + "} break;\n"
                    dbase = dbase + "\tcase " + str(i) + ": {" + \
                        "{" + pretty_print.C99_print(simplify(diff(fe.N[real_index], x))).replace(" = 0;", ".setZero();").replace(" = 1;", ".setOnes();").replace(" = -1;", ".setConstant(-1);") + "val.col(0) = result_0; }" \
                        "{" + pretty_print.C99_print(simplify(diff(fe.N[real_index], y))).replace(" = 0;", ".setZero();").replace(" = 1;", ".setOnes();").replace(" = -1;", ".setConstant(-1);") + "val.col(1) = result_0; }"

                dbase = dbase + "} break;\n"

            base = base + "\tdefault: assert(false);\n}"
            dbase = dbase + "\tdefault: assert(false);\n}"

            cppv = cppv + func + "{\n\n"
            cppv = cppv + base + "}\n"

            cppg = cppg + dfunc + "{\n\n"
            cppg = cppg + dbase + "}\n\n"
            cppn = cppn + nodes + "\n\n"

            if dim == 3:
                with open(os.path.join(path, f"{nameg}_{order}.cpp"), "w") as file:
                    file.write(cppg+"}}")
                    cppg="#include <Eigen/Dense>\n namespace polyfem {\nnamespace autogen {"

        if dim == 3:
            cppg=""
        unique_nodes = unique_nodes + "\tdefault: assert(false);\n}}"

        unique_fun = unique_fun + "\tdefault: assert(false);\n}}"
        dunique_fun = dunique_fun + "\tdefault: assert(false);\n}}"

        cppv = cppv + "}\n\n"
        cppn = cppn + "}\n\n"
        if dim != 3:
            cppg = cppg + "}\n\n"

        cppn = cppn + unique_nodes + "\n}}\n"
        cppv = cppv + unique_fun + "\n}}\n"
        cppg = cppg + dunique_fun + "\n}}\n"
        hppv = hppv + "\n}}\n"
        hppn = hppn + "\n}}\n"
        hppg = hppg + "\n}}\n"

        if dim == 3:
            tcppg = f"#include \"{nameg}.hpp\"\n\n\n"
            tcppg = tcppg + "namespace polyfem {\nnamespace autogen {\n"
            tcppg = tcppg + eextern + "\n"
            cppg=tcppg+cppg

        print("saving...")
        with open(os.path.join(path, f"{namev}.cpp"), "w") as file:
            file.write(cppv)
        with open(os.path.join(path, f"{namen}.cpp"), "w") as file:
            file.write(cppn)
        with open(os.path.join(path, f"{nameg}.cpp"), "w") as file:
            file.write(cppg)

        with open(os.path.join(path, f"{namev}.hpp"), "w") as file:
            file.write(hppv)
        with open(os.path.join(path, f"{namen}.hpp"), "w") as file:
            file.write(hppn)
        with open(os.path.join(path, f"{nameg}.hpp"), "w") as file:
            file.write(hppg)


    hpp = "#pragma once\n\n#include <Eigen/Dense>\n\n"
    for dim in dims:
        hpp = hpp + f"#include \"auto_q_bases_{dim}d_val.hpp\"\n"
        hpp = hpp + f"#include \"auto_q_bases_{dim}d_nodes.hpp\"\n"
        hpp = hpp + f"#include \"auto_q_bases_{dim}d_grad.hpp\"\n"
    hpp = hpp + "\n\nnamespace polyfem {\nnamespace autogen " + "{\n"
    hpp = hpp + "\nstatic const int MAX_Q_BASES = " + str(max(orders)) + ";\n"
    hpp = hpp + "\n}}\n"



    print("saving...")
    with open(os.path.join(path, "auto_q_bases.hpp"), "w") as file:
        file.write(hpp)

    print("done!")
