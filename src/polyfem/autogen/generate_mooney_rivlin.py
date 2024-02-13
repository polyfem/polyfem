from sympy import symbols, trace, det, log, Pow, Rational, MatrixSymbol, Matrix, ccode
from pretty_print import C99_print_tensor
import re

c1, c2, c3, d1 = symbols("c1 c2 c3 d1")


def optimizations(s):
    ret = s.replace("[", "(").replace("]", ")")

    def replace_pow_with_x_times_x(match):
        return match.group(1) + '*' + match.group(1)
    # Regular expression pattern to match "pow(x, 2)" where x is a variable
    pattern = r'pow\(([^,]+), 2\)'
    # Use re.sub to replace all occurrences
    ret = re.sub(pattern, replace_pow_with_x_times_x, ret)
    return ret


def mr_energy(F_):
    # Right cauchy green tensor
    RCG = F_ @ F_.T
    def first(X): return Pow(det(F_), Rational(-2, dim)) * trace(X)

    def second(X): return Pow(det(F_), Rational(-4, dim)) * \
        Rational(1, 2) * (trace(X)*trace(X) - trace(X*X))
    energy = c1 * (first(RCG) - dim) + c2 * (second(RCG) - dim) + c3 * \
        (first(RCG) - dim) * (second(RCG) - dim) + \
        d1 * log(det(F_)) * log(det(F_))
    return energy


print(R"""
#include "auto_mooney_rivlin_gradient_hessian.hpp"

namespace polyfem {
        namespace autogen {


template<>
void generate_gradient_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 2, 2> &def_grad, Eigen::Matrix<double, 2, 2> &gradient)
{
""")


dim = 2
F = MatrixSymbol('def_grad', dim, dim)
F_ = Matrix(F)

energy = mr_energy(F_)
grad = energy.diff(F_)
hess = grad.diff(F_)

print(optimizations(C99_print_tensor(grad, "gradient")))


print(R"""
}

template <>
void generate_hessian_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 2, 2> &def_grad, Eigen::Matrix<double, 4, 4> &hessian)
{
""")

print(optimizations(C99_print_tensor(hess, "hessian")))

print(R"""
}

template <>
void generate_gradient_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 3, 3> &def_grad, Eigen::Matrix<double, 3, 3> &gradient){

""")

dim = 3
F = MatrixSymbol('def_grad', dim, dim)
F_ = Matrix(F)

energy = mr_energy(F_)
grad = energy.diff(F_)
hess = grad.diff(F_)

print(optimizations(C99_print_tensor(grad, "gradient")))

print(R"""
}

template <>
void generate_hessian_(const double c1, const double c2, const double c3, const double d1, const Eigen::Matrix<double, 3, 3> &def_grad, Eigen::Matrix<double, 9, 9> &hessian)
{
""")

print(optimizations(C99_print_tensor(hess, "hessian")))

print(R"""
}

void generate_gradient(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &def_grad, Eigen::MatrixXd &gradient)
{
        int dim = def_grad.rows();

if (dim == 2)
{
	Eigen::Matrix<double, 2, 2> temp;
	generate_gradient_<2>(c1, c2, c3, d1, def_grad, temp);
	gradient = temp;
}
if (dim == 3)
{
	Eigen::Matrix<double, 3, 3> temp;
	generate_gradient_<3>(c1, c2, c3, d1, def_grad, temp);
	gradient = temp;
}
}

void generate_hessian(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &def_grad, Eigen::MatrixXd &hessian)
{
        int dim = def_grad.rows();

if (dim == 2)
{
	Eigen::Matrix<double, 4, 4> temp;
	generate_hessian_<2>(c1, c2, c3, d1, def_grad, temp);
	hessian = temp;
}
if (dim == 3)
{
	Eigen::Matrix<double, 9, 9> temp;
	generate_hessian_<3>(c1, c2, c3, d1, def_grad, temp);
	hessian = temp;
}
}
}
}
      """)
