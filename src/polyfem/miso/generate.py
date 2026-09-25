"""
Generate miso C++ solver classes for polyfem Jacobian validity checks.

Produces two classes per (D, P) pair:
  P{P}{Simplex}Val  -- SOLVE: static injectivity check (det J > 0 everywhere)
  P{P}{Simplex}CGV  -- MINIMIZE: continuous geometric validity (max safe time step)
  JacEval_P{P}{Simplex}  -- evaluator for det J at a point (used for P=1 static check
                             and robust_evaluate_jacobian)

Run:
    python generate.py
"""
import os
import sys
import shutil
import subprocess

from sympy import Matrix, symbols

# Clone pymiso from GitLab into ignore/pymiso if not already present
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYMISO_DIR = os.path.join(_HERE, 'ignore', 'pymiso')
_PYMISO_URL = 'https://gitlab.com/minimize-solve/pymiso.git'

if not os.path.isdir(_PYMISO_DIR):
    os.makedirs(os.path.join(_HERE, 'ignore'), exist_ok=True)
    subprocess.run(['git', 'clone', _PYMISO_URL, _PYMISO_DIR], check=True)

sys.path.insert(0, _PYMISO_DIR)
from miso import Domain, Basis, make_poly, generate, generate_evaluator

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')

SIMPLEX = {1: 'Lin', 2: 'Tri', 3: 'Tet'}


def SimplexChecks(D, P):
    """
    Generate both static (Val) and continuous (CGV) validity solvers for a
    degree-P map on a D-simplex. Two generate() calls share the symbolic setup.

    Control points are rest position + displacement, taken as separate inputs
    (r, u) so the sum is formed in interval arithmetic: the displacement is the
    solver's own vector, which the continuous check widens to cover the step
    the solver actually takes (fix.pdf), and nothing rounds it on the way in.
    """
    simplex = SIMPLEX[D]
    X = Domain(' '.join(f'x{i}' for i in range(D)))
    T = Domain('t')
    domain = X * T
    t = domain.variables[-1]  # T[0] may differ by identity from domain.variables

    def rest_plus(name):
        r = X.poly_map(P, Basis.LAGRANGE, codomain_dim=D, name='r', output_miso_poly=False)
        u = X.poly_map(P, Basis.LAGRANGE, codomain_dim=D, name=name, output_miso_poly=False)
        return r + u

    # Compute the static Jacobian determinant (shared symbolic work)
    x = rest_plus('u')
    Jx = x.jacobian(X.variables).applyfunc(make_poly)
    jd_val = make_poly(Jx.det())

    # Evaluator: standalone function computing det J at a point (all free symbols
    # become parameters). Used for P=1 static validity check (constant det) and
    # for robust_evaluate_jacobian (pointwise evaluation for P>1).
    generate_evaluator(SRC_DIR, f'JacEval_P{P}{simplex}', jd_val)
    print(f'JacEval_P{P}{simplex} done')

    # Call 1: static validity (SOLVE, no objective)
    # P=1: Jacobian of a linear map is constant in domain variables (degree 0);
    # miso's Bernstein converter requires degree >= 1.  Handled directly in C++.
    if P > 1:
        generate(SRC_DIR, f'P{P}{simplex}Val', X, [jd_val], parallel=True)
        print(f'P{P}{simplex}Val done')

    # CGV: blended map (1-t)*p0 + t*p1 with p0 = r + u0, p1 = r + u1
    p0 = rest_plus('u0')
    p1 = rest_plus('u1')

    if P == 1:
        # The Jacobian of a linear map is constant in space, so validity
        # along the step is a one-variable problem in t.
        tt = T.variables[0]
        Jt = p0.jacobian(X.variables) * (1 - tt) + p1.jacobian(X.variables) * tt
        jd_t = make_poly(Jt.det())
        generate(SRC_DIR, f'P{P}{simplex}CGV', T, [jd_t], objective=tt, parallel=True)
        print(f'P{P}{simplex}CGV done')
        return

    p = [make_poly(a * (1 - t) + b * t) for a, b in zip(p0, p1)]
    p_ext = p + [t]  # plain symbol: avoids sympy Function canonicalization swapping t identity
    jd_cgv = make_poly(Matrix(p_ext).jacobian(domain.variables).applyfunc(make_poly).det())
    sd_all = domain.default_subdivision
    sd_t = domain.make_subdivision(T)

    # Call 2: continuous geometric validity (MINIMIZE, objective=t)
    generate(SRC_DIR, f'P{P}{simplex}CGV', domain, [jd_cgv],
             objective=t, subdivisions=[sd_all, sd_t], parallel=True)
    print(f'P{P}{simplex}CGV done')


if __name__ == '__main__':
    os.makedirs(SRC_DIR, exist_ok=True)
    # Clear existing class subdirectories; keep top-level CMakeLists.txt / include.hpp
    for entry in os.scandir(SRC_DIR):
        if entry.is_dir():
            shutil.rmtree(entry.path)

    for D, P in [(2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3)]:
        SimplexChecks(D, P)

    print('All done.')
