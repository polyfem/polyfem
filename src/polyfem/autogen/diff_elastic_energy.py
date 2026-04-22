import sympy as sp
from sympy.printing import ccode
import pretty_print
import argparse
import os
import numpy as np
from pathlib import Path


class AMIPSEnergy:
    def __init__(self, use_rest_pose, dim):
        self.use_rest_pose = use_rest_pose
        self.dim = dim

    def parameters(self):
        return []

    def get_standard(self):
        if self.use_rest_pose:
            return sp.eye(self.dim)

        if self.dim == 2:
            standard = sp.Matrix([
                [1, 0],
                [sp.Rational(1, 2), sp.sqrt(3) / 2]
            ])
        else:
            standard = sp.Matrix([
                [1, 0, 0],
                [sp.Rational(1, 2), sp.sqrt(3) / 2, 0],
                [sp.Rational(1, 2),
                 sp.Rational(1, 2) / sp.sqrt(3), sp.sqrt(3) / 2]
            ])
        standard = standard.inv().T

        return standard

    def eval(self, p, t, el_id, def_grad):
        if self.use_rest_pose:
            power = 1 if self.dim == 2 else sp.Rational(2, 3)
        else:
            power = 2 if self.dim == 2 else sp.Rational(5, 3)

        standard = self.get_standard()

        if not self.use_rest_pose:
            def_grad = def_grad @ standard

        det = def_grad.det()

        # if det <= 0:
        #     return sp.nan

        powJ = det**power
        return (def_grad.T @ def_grad).trace() / powJ


class VolumePenaltyEnergy:
    def __init__(self, dim):
        self.dim = dim
        self.k_ = sp.Symbol('k')

    def parameters(self):
        return ["k"]

    def eval(self, p, t, el_id, def_grad):
        k = self.k_

        J = def_grad.det()
        log_J = sp.log(J)

        val = k / 2.0 * ((J * J - 1) / 2.0 - log_J)

        return val


def compute_energy(energy):
    dim = energy.dim
    F = sp.Matrix(dim, dim, lambda i, j: sp.Symbol(f"F{i}{j}"))
    p = sp.MatrixSymbol('p', dim, 1)
    t = sp.Symbol('t')
    el_id = sp.Symbol('el_id')

    E = energy.eval(p, t, el_id, F)
    E = sp.simplify(E)

    x = sp.Matrix([F[i, j] for i in range(dim) for j in range(dim)])
    grad = sp.Matrix([sp.diff(E, v) for v in x])
    hess = grad.jacobian(x)

    return E, grad, hess


energies = {
    "VolumePenalty2d": VolumePenaltyEnergy(dim=2),
    "VolumePenalty3d": VolumePenaltyEnergy(dim=3),
    "AMIPS2d": AMIPSEnergy(use_rest_pose=False, dim=2),
    "AMIPS2drest": AMIPSEnergy(use_rest_pose=True, dim=2),
    "AMIPS3d": AMIPSEnergy(use_rest_pose=False, dim=3),
    "AMIPS3drest": AMIPSEnergy(use_rest_pose=True, dim=3),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=str, help="path to the output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = Path(os.path.abspath(args.output))

    for name, energy in energies.items():
        E, grad, hess = compute_energy(energy)
        dim = energy.dim

        file = path / f"{name}.hpp"

        fparams = "(const RowVectorNd &p, const double t, const int el_id, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>& F"
        for param in energy.parameters():
            fparams += f", const double {param}"
        fparams += ")"

        with open(file, "w") as f:
            f.write(f"// Auto-generated code for {name} energy\n")
            f.write(f"#pragma once\n")
            f.write("#include <Eigen/Dense>\n\n")
            f.write("namespace polyfem {\n")
            f.write(f"    namespace autogen {{\n")

            # f.write(f"        inline double {name}_energy{fparams} {{\n")
            # for i in range(dim):
            #     for j in range(dim):
            #         f.write(f"            const double F{i*dim + j} = F({i}, {j});\n")
            # f.write(f"            const double {pretty_print.C99_print(sp.sympify(E))}\n")
            # f.write(f"            return result_0;\n")
            # f.write("        }\n\n")

            f.write(
                f"        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> {name}_gradient{fparams} {{\n")
            f.write(
                f"            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad({dim},{dim});\n")
            for i in range(dim):
                for j in range(dim):
                    f.write(
                        f"            const double F{i}{j} = F({i}, {j});\n")
            f.write(f"            std::array<double, {dim*dim}> result_0;\n")
            f.write(
                f"            {pretty_print.C99_print(sp.sympify(grad))};\n")
            for i in range(dim):
                for j in range(dim):
                    idx = i * dim + j
                    f.write(f"            grad({i}, {j}) = result_0[{idx}];\n")
            f.write("            return grad;\n")
            f.write("        }\n\n")

            f.write(
                f"        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> {name}_hessian{fparams} {{\n")
            f.write(
                f"            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess({dim*dim},{dim*dim});\n")
            f.write(f"            std::array<double, {dim**4}> result_0;\n")
            for i in range(dim):
                for j in range(dim):
                    f.write(
                        f"            const double F{i}{j} = F({i}, {j});\n")
            f.write(
                f"            {pretty_print.C99_print(sp.sympify(hess))};\n")
            for i in range(dim*dim):
                for j in range(dim*dim):
                    f.write(
                        f"            hess({i}, {j}) = result_0[{i*dim*dim+j}];\n")
            f.write("            return hess;\n")
            f.write("        }\n")
            f.write("    }\n")
            f.write("}\n")
