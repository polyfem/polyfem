from sympy import *
from sympy.matrices import *
from sympy.printing import ccode
import numpy as np


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
