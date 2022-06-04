# -*- coding: utf-8 -*-

import math
import textwrap
import numpy
import sympy
import quadpy


def integrate_exact(f):
    """
    Integrate a function over the reference tetrahedron (0,0), (1,0), (0,1).
    """
    x, y, z = sympy.symbols('x y z')
    exact = sympy.integrate(
        sympy.integrate(f, (y, 0, 1 - x)),
        (x, 0, 1)
    )
    exact = sympy.integrate(
        sympy.integrate(
            sympy.integrate(f, (z, 0, 1 - x - y)),
            (y, 0, 1 - x)
        ),
        (x, 0, 1)
    )
    return exact


def integrate_approx(f, scheme):
    """
    Integrate numerically over the reference tetrahedron.
    """
    x, y, z = sympy.symbols('x y z')
    vol = 1.0 / 6.0  # volume of the reference tetrahedron
    res = 0
    for i, w in enumerate(scheme.weights):
        res += w * f.subs([(x, scheme.points[i, 0]),
                           (y, scheme.points[i, 1]), (z, scheme.points[i, 2])])
    return vol * res


def list_schemes():
    """
    List all existing schemes for tetrahedra.
    """
    L = [quadpy.tetrahedron.BeckersHaegemans(k) for k in [8, 9]] \
        + [quadpy.tetrahedron.Gatermann()] \
        + [quadpy.tetrahedron.GrundmannMoeller(k) for k in range(8)] \
        + [quadpy.tetrahedron.HammerStroud(k) for k in [2, 3]] \
        + [quadpy.tetrahedron.HammerMarloweStroud(k) for k in [1, 2, 3]] \
        + [quadpy.tetrahedron.Keast(k) for k in range(11)] \
        + [quadpy.tetrahedron.LiuVinokur(k) for k in range(1, 15)] \
        + [quadpy.tetrahedron.MaeztuSainz()] \
        + [quadpy.tetrahedron.NewtonCotesClosed(k) for k in range(1, 7)] \
        + [quadpy.tetrahedron.NewtonCotesOpen(k) for k in range(7)] \
        + [quadpy.tetrahedron.ShunnHam(k) for k in range(1, 7)] \
        + [quadpy.tetrahedron.Stroud(k) for k in ['T3 5-1', 'T3 7-1']] \
        + [quadpy.tetrahedron.VioreanuRokhlin(k) for k in range(10)] \
        + [quadpy.tetrahedron.Walkington(k) for k in [1, 2, 3, 5, 'p5', 7]] \
        + [quadpy.tetrahedron.WilliamsShunnJameson()] \
        + [quadpy.tetrahedron.WitherdenVincent(k) for k in [1, 2, 3, 5, 6, 7, 8, 9, 10]] \
        + [quadpy.tetrahedron.XiaoGimbutas(k) for k in range(1, 16)] \
        + [quadpy.tetrahedron.Yu(k) for k in range(1, 6)] \
        + [quadpy.tetrahedron.ZhangCuiLiu(k) for k in [1, 2]] \
        + [quadpy.tetrahedron.Zienkiewicz(k) for k in [4, 5]]

    return L


def generate_monomials(order):
    """
    Generate trivariate monomials up to given order.
    """
    monoms = []
    x, y, z = sympy.symbols('x y z')
    for i in range(0, order + 1):
        for j in range(0, order + 1):
            for k in range(0, order + 1):
                if i + j + k <= order:
                    monoms.append(x**i * y**j * z**k)
    return monoms


def is_valid(scheme, tol=1e-6, relaxed=False):
    """
    A scheme is valid if:
    1. All its weights sums up to one;
    2. All the weights are positive;
    3. All its points are inside the reference tetrahedron;
    4. No point lie on an face.
    """
    return math.isclose(numpy.sum(scheme.weights), 1.0, rel_tol=1e-10) \
        and (relaxed or (scheme.weights >= tol).all()) \
        and (scheme.points >= 0).all() \
        and (scheme.points[:, 0] + scheme.points[:, 1] + scheme.points[:, 2] <= 1 - tol).all() \
        and (scheme.points[:, 0] >= tol).all() \
        and (scheme.points[:, 1] >= tol).all() \
        and (scheme.points[:, 2] >= tol).all()


def pick_scheme(all_schemes, order, relaxed=False):
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
    for scheme in all_schemes:
        if scheme.degree == order and is_valid(scheme, relaxed=relaxed):
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
    return best_scheme, best_error


def generate_cpp(selected_schemes):
    """
    Generate cpp code to fill points & weights.
    """
    res = []
    for order, scheme in selected_schemes:
        try:
            name = scheme.name
        except AttributeError:
            name = scheme.__class__.__name__
        code = """\
        case {order}: // {name}
            points.resize({num_pts}, 3);
            weights.resize({num_pts}, 1);
            points << {points};
            weights << {weights};
            break;
        """.format(order=order,
                   name=name,
                   num_pts=len(scheme.weights),
                   points=', '.join('{:.64g}'.format(w)
                                    for w in scheme.points.flatten()),
                   weights=', '.join('{:.64g}'.format(w) for w in scheme.weights))
        res.append(textwrap.dedent(code))
    return ''.join(res)


def main():
    all_schemes = list_schemes()
    selected = []
    for order in range(1, 16):
        print("order", order)
        scheme, error = pick_scheme(all_schemes, order)
        if scheme is None:
            scheme, error = pick_scheme(all_schemes, order, relaxed=True)
            assert scheme is not None
        print(error)
        selected.append((order, scheme))
    code = generate_cpp(selected)
    with open('../auto_tetrahedron.ipp', 'w') as f:
        f.write(code)


if __name__ == '__main__':
    main()
