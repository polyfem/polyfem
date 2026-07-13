from sympy import *
from sympy.matrices import *
from sympy.printing import ccode
import numpy as np

# We define reference space coordinate as sympy symbol x, y, z.
SCALAR_COORDS = symbols('x,y,z')

# pretty print
def C99_print(expr):
    CSE_results = cse(expr, numbered_symbols("helper_"), optimizations='basic')
    lines = []
    for helper in CSE_results[0]:
        if isinstance(helper[1], MatrixSymbol):
            lines.append(
                'const auto ' + str(helper[0]) + '[' + str(helper[1].rows * helper[1].cols) + '];')
            lines.append(ccode(helper[1], helper[0]))
        else:
            lines.append('const auto ' + ccode(helper[1], helper[0]))

    for i, result in enumerate(CSE_results[1]):
        lines.append(ccode(result, "result_%d" % i))
    return '\n'.join(lines)


def C99_print_scalar(expr, result_name="result"):
    """Print scalar assignment: double helper_x = expr;"""
    substitutions, results = cse(
        expr, numbered_symbols("helper_"), optimizations='basic')
    lines = [f"double {ccode(value, symbol)}" for symbol, value in substitutions]
    lines.append(ccode(results[0], result_name))
    return '\n'.join(lines)


def scalar_args(dim):
    """Return a scalar function argument list, e.g. 'double x, double y'."""
    assert 1 <= dim <= len(SCALAR_COORDS)
    return ", ".join(f"double {coord.name}" for coord in SCALAR_COORDS[:dim])


def scalar_call_args(dim):
    """
    Return scalar function call args: x[i], y[i], z[i].
    This is for unpacking quadrature points in separate coordinate spans.
    """
    return ", ".join(f"{SCALAR_COORDS[d].name}[i]" for d in range(dim))


def C99_print_scalar_value_function(function_name, expr, dim):
    """Print function that evaluate basis value at one quadrature point."""
    return (
        f"inline POLYFEM_BOTH double {function_name}({scalar_args(dim)}) {{\n"
        "double result;\n"
        f"{C99_print_scalar(expr, 'result')}\n"
        "return result;\n"
        "}\n\n")


def C99_print_scalar_gradient_function(function_name, expr, dim):
    """Print function that evaluate basis gradient at one quadrature point."""
    assert 1 <= dim <= len(SCALAR_COORDS)
    lines = [f"inline POLYFEM_BOTH void {function_name}({scalar_args(dim)}, double *val) {{"]
    for d, coord in enumerate(SCALAR_COORDS[:dim]):
        derivative = simplify(diff(expr, coord))
        lines.append("{" + C99_print_scalar(derivative, f"val[{d}]") + "}")
    lines.append("}\n")
    return "\n".join(lines) + "\n"


def C99_print_scalar_value_case(local_index, function_name, dim):
    """Generate one local_index switch case for basis values function."""
    return (
        f"\tcase {local_index}:\n"
        "\t\tfor (std::size_t i = 0; i < x.size(); ++i)\n"
        f"\t\t\tval[i] = {function_name}({scalar_call_args(dim)});\n"
        "\t\tbreak;\n")


def C99_print_scalar_gradient_case(local_index, function_name, dim):
    """Generate one local_index switch case for basis gradients."""
    outputs = ["grad_x", "grad_y", "grad_z"]
    lines = [
        f"\tcase {local_index}:",
        "\t\tfor (std::size_t i = 0; i < x.size(); ++i) {",
        f"\t\t\t{function_name}({scalar_call_args(dim)}, gradient);",
    ]
    lines.extend(f"\t\t\t{outputs[d]}[i] = gradient[{d}];" for d in range(dim))
    lines.extend(["\t\t}", "\t\tbreak;"])
    return "\n".join(lines) + "\n"


# Pretty print a matrix or tensor expression.
def C99_print_tensor(expr, result_name="result"):
    # If a tensor expression, the result is reshaped into a 2d matrix for printing.
    lines = []
    subs, result = cse(expr, numbered_symbols(
        "helper_"), optimizations='basic')
    if len(result) == 1:
        result = result[0]

    for k, v in subs:
        lines.append(f"const double {ccode(v, k)}")

    result_shape = np.array(result).shape
    if len(result_shape) == 2:
        for i in range(result_shape[0]):
            for j in range(result_shape[1]):
                s = ccode(result[i, j], f"{result_name}[{i}, {j}]")
                lines.append(f"{s}")
    elif len(result_shape) == 4:
        for i in range(result_shape[0]):
            for j in range(result_shape[1]):
                for k in range(result_shape[2]):
                    for l in range(result_shape[3]):
                        s = ccode(result[i, j, k, l],
                                  f"{result_name}[{i * result_shape[1] + j}, {k * result_shape[3] + l}]")
                        lines.append(f"{s}")

    return "\n".join(lines)
