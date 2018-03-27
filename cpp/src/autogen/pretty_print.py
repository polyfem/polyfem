from sympy import *
from sympy.matrices import *
from sympy.printing import ccode


# pretty print
def C99_print(expr):
    CSE_results = cse(expr, numbered_symbols("helper_"), optimizations='basic')
    lines = []
    for helper in CSE_results[0]:
        if isinstance(helper[1], MatrixSymbol):
            lines.append('const auto ' + str(helper[0]) + '[' + str(helper[1].rows * helper[1].cols) + '];')
            lines.append(ccode(helper[1], helper[0]))
        else:
            lines.append('const auto ' + ccode(helper[1], helper[0]))

    for i, result in enumerate(CSE_results[1]):
        lines.append(ccode(result, "result_%d" % i))
    return '\n'.join(lines)
