# -*- coding: utf-8 -*-

import math
import textwrap
import numpy
import sympy
import quadpy


def integrate_exact(f):
    """
    Integrate a function over the reference triangle (0,0), (1,0), (0,1).
    """
    x, y = sympy.symbols('x y')
    exact = sympy.integrate(
        sympy.integrate(f, (y, 0, 1 - x)),
        (x, 0, 1)
    )
    return exact


def integrate_approx(f, scheme):
    """
    Integrate numerically over the reference triangle.
    """
    x, y = sympy.symbols('x y')
    vol = 0.5  # area of the reference triangle
    res = 0
    for i, w in enumerate(scheme.weights):
        res += w * f.subs([(x, scheme.points[i, 0]), (y, scheme.points[i, 1])])
    return vol * res


def list_schemes():
    """
    List all existing schemes for triangles.
    """
    L = [(quadpy.triangle.BerntsenEspelid(k), 1.0e-11) for k in range(1, 5)] \
        + [(quadpy.triangle.Centroid(), 1.0e-14)] \
        + [(quadpy.triangle.CoolsHaegemans(k), 1.0e-13) for k in [1]] \
        + [(quadpy.triangle.Cubtri(), 1.0e-14)] \
        + [(quadpy.triangle.Dunavant(k), 1.0e-12) for k in range(1, 21)] \
        + [(quadpy.triangle.Gatermann(), 1.0e-12)] \
        + [(quadpy.triangle.GrundmannMoeller(k), 1.0e-12) for k in range(10)] \
        + [(quadpy.triangle.HammerMarloweStroud(k), 1.0e-14) for k in range(1, 6)] \
        + [(quadpy.triangle.HammerStroud(k), 1.0e-14) for k in [2, 3]] \
        + [(quadpy.triangle.Hillion(k), 1.0e-14) for k in range(1, 11)] \
        + [(quadpy.triangle.LaursenGellert(key), 1.0e-13) for key in quadpy.triangle.LaursenGellert.keys] \
        + [(quadpy.triangle.Lether(k), 1.0e-14) for k in range(1, 14)] \
        + [(quadpy.triangle.LiuVinokur(k), 1.0e-14) for k in range(1, 14)] \
        + [(quadpy.triangle.LynessJespersen(k), 1.0e-11) for k in range(1, 22)] \
        + [(quadpy.triangle.NewtonCotesClosed(k), 1.0e-14) for k in range(1, 6)] \
        + [(quadpy.triangle.NewtonCotesOpen(k), 1.0e-13) for k in range(6)] \
        + [(quadpy.triangle.Papanicolopulos('fs', k), 1.0e-13) for k in range(9)] \
        + [(quadpy.triangle.Papanicolopulos('rot', k), 1.0e-14)
           # The first 8 schemes are flawed by round-off error
           for k in range(8, 18)] \
        + [(quadpy.triangle.SevenPoint(), 1.0e-14)] \
        + [(quadpy.triangle.Strang(k), 1.0e-14) for k in range(1, 11)] \
        + [(quadpy.triangle.Stroud(k), 1.0e-12) for k in ['T2 3-1', 'T2 5-1', 'T2 7-1']] \
        + [(quadpy.triangle.TaylorWingateBos(k), 1.0e-12) for k in [1, 2, 4, 5, 8]] \
        + [(quadpy.triangle.Triex(k), 1.0e-13) for k in [19, 28]] \
        + [(quadpy.triangle.Vertex(), 1.0e-14)] \
        + [(quadpy.triangle.VioreanuRokhlin(k), 1.0e-11) for k in range(20)] \
        + [(quadpy.triangle.Walkington(k), 1.0e-14) for k in [1, 2, 3, 5, 'p5']] \
        + [(quadpy.triangle.WandzuraXiao(k), 1.0e-14) for k in range(1, 7)] \
        + [(quadpy.triangle.WilliamsShunnJameson(k), 1.0e-11) for k in range(1, 9)] \
        + [(quadpy.triangle.WitherdenVincent(k), 1.0e-14) for k in [
            1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        ]] \
        + [(quadpy.triangle.XiaoGimbutas(k), 1.0e-14) for k in range(1, 51)] \
        + [(quadpy.triangle.ZhangCuiLiu(k), 1.0e-14) for k in [1, 2, 3]]

    return L


def generate_monomials(order):
    """
    Generate bivariate monomials up to given order.
    """
    monoms = []
    x, y = sympy.symbols('x y')
    for i in range(0, order + 1):
        for j in range(0, order + 1):
            if i + j <= order:
                monoms.append(x**i * y**j)
    return monoms


def is_valid(scheme, tol=1e-6):
    """
    A scheme is valid if:
    1. All its weights sums up to one;
    2. All the weights are positive;
    3. All its points are inside the reference triangle;
    4. No point lie on an edge.
    """
    return math.isclose(numpy.sum(scheme.weights), 1.0, rel_tol=1e-10) \
        and (scheme.weights >= tol).all() \
        and (scheme.points >= 0).all() \
        and (scheme.points[:, 0] + scheme.points[:, 1] <= 1 - tol).all() \
        and (scheme.points[:, 0] >= tol).all() \
        and (scheme.points[:, 1] >= tol).all()


def pick_scheme(all_schemes, order):
    """
    Picks the best scheme for a given polynomial degree, following this strategy:
    - Tries all schemes with the same order as required.
    - Eliminate schemes with weights that do not sum up to 1, or points that lie
      outside the reference triangle.
    - Among the remaining schemes, compare the integration error over all bivariate
      monomials of inferior order.
    - Keeps only the scheme with total integration error lower than a certain threshold.
    - Among those, keep the one with fewer number of integration points.
    - Finally, break ties using the integration error accumulated over the monomials.
    """
    monoms = generate_monomials(order)
    print(monoms)
    min_points = None
    best_error = None
    best_scheme = None
    threshold = 1e-12  # Only accept quadrature rules with this much precision
    for scheme, __ in all_schemes:
        if scheme.degree == order and is_valid(scheme):
            N = len(scheme.weights)
            ok = True
            error = 0
            for poly in monoms:
                exact = integrate_exact(poly)
                approx = integrate_approx(poly, scheme)
                # print(abs(exact - approx), tol)
                if not math.isclose(exact - approx, 0, abs_tol=threshold):
                    ok = False
                error += abs(exact - approx)
            if ok:
                if (min_points is None) or N < min_points:
                    min_points = N
                    best_error = error
                    best_scheme = scheme
                elif N == min_points and error < best_error:
                    best_error = error
                    best_scheme = scheme
    return best_scheme


def generate_cpp(selected_schemes):
    """
    Generate cpp code to fill points & weights.
    """
    res = []
    for order, scheme in selected_schemes:
        code = """\
        case {order}: // {name}
            points.resize({num_pts}, 2);
            weights.resize({num_pts}, 1);
            points << {points};
            weights << {weights};
            break;
        """.format(order=order,
                   name=scheme.name,
                   num_pts=len(scheme.weights),
                   points=', '.join('{:.64g}'.format(w) for w in scheme.points.flatten()),
                   weights=', '.join('{:.64g}'.format(w) for w in scheme.weights))
        res.append(textwrap.dedent(code))
    return ''.join(res)


def main():
    all_schemes = list_schemes()
    selected = []
    for order in range(1, 16):
        scheme = pick_scheme(all_schemes, order)
        selected.append((order, scheme))
    code = generate_cpp(selected)
    # print(code)
    with open('../auto_triangle.ipp', 'w') as f:
        f.write(code)


if __name__ == '__main__':
    main()
