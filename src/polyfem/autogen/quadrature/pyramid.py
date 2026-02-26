# -*- coding: utf-8 -*-
import math
import textwrap
import numpy as np
import sympy
# https://github.com/yib0liu/legacy-quadpy commit 5213fed9155e3cf0e62900a7483ecf9abb23d48a
import quadpy

b = np.array([
    [0,0,0],
    [0,1,0],
    [1,1,0],
    [1,0,0],
    [0,0,1]
])

def integrate_exact(f):
    """
    Integrate a function over the reference pyramid:
        0 <= z <= 1
        0 <= x <= (1 - z)
        0 <= y <= (1 - z)
    """
    import sympy
    x, y, z = sympy.symbols('x y z')

    exact = sympy.integrate(
        sympy.integrate(
            sympy.integrate(f, (x,0, (1 - z))),
            (y, 0, (1 - z)))
        , (z, 0, 1)
    )
    return exact



def integrate_approx(f, scheme):
    """
    Integrate numerically over the reference tetrahedron.
    """
    x, y, z = sympy.symbols('x y z')
    res = 0
    pts=scheme.transform(b)
    weights=scheme.transform_w(b)
    for i, w in enumerate(weights):
        res += w * f.subs([(x, pts[i, 0]),
                           (y, pts[i, 1]), (z, pts[i, 2])])
    return  res


def list_schemes():
    """
    List all existing schemes for tetrahedra.
    """
    L = [quadpy.p3.felippa_1()] \
        + [quadpy.p3.felippa_2()] \
        + [quadpy.p3.felippa_3()] \
        + [quadpy.p3.felippa_4()] \
        + [quadpy.p3.felippa_5()] \
        + [quadpy.p3.felippa_6()] \
        + [quadpy.p3.felippa_7()] \
        + [quadpy.p3.felippa_8()] \
        + [quadpy.p3.felippa_9()] \

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
    1. weights sum to 1
    2. weights positive
    3. points inside reference pyramid:
         0 <= z <= 1
         |x|, |y| <= 1 - z
    """
    pts=scheme.transform(b)
    w=scheme.transform_w(b)
    print("sum w: ", sum(w))

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    cond_z = (z >= 0 - tol) & (z <= 1 + tol)
    cond_x = (x >= 0 - tol) & (x <= (1 - z) + tol)
    cond_y = (y >= 0 - tol) & (y <= (1 - z) + tol)
    cond_inside = cond_z & cond_x & cond_y

    print("is valid ",(
        math.isclose(np.sum(w), 0.3333333333333333, rel_tol=1e-10)
        ,(relaxed or (w >= tol).all())
        # , cond_z.all() and cond_x.all() and cond_y.all()
        ,cond_inside.all()
    ))
    
    return (
        math.isclose(np.sum(w),  0.3333333333333333, rel_tol=1e-10)
        and (relaxed or (w >= tol).all())
        # and cond_z.all() and cond_x.all() and cond_y.all()
        and cond_inside.all()
    )

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
        points=scheme.transform(b)
        weights=scheme.transform_w(b)
        code = """\
        case {order}: // {name}
            points.resize({num_pts}, 3);
            weights.resize({num_pts}, 1);
            points << {points};
            weights << {weights};
            break;
        """.format(order=order,
                   name=name,
                   num_pts=len(weights),
                   points=', '.join('{:.64g}'.format(w)
                                    for w in points.flatten()),
                   weights=', '.join('{:.64g}'.format(w) for w in weights))
        res.append(textwrap.dedent(code))
    return ''.join(res)


def main():
    all_schemes = list_schemes()
    selected = []
    for order in [1,2,3,5]:  # legacy-quadpy only support these
        scheme, error = pick_scheme(all_schemes, order)
        if scheme is None:
            scheme, error = pick_scheme(all_schemes, order, relaxed=True)
            # assert scheme is not None
        if scheme is not None:
            selected.append((order, scheme))
    code = generate_cpp(selected)
    with open('../auto_pyramid.ipp', 'w') as f:
        f.write(code)
        print("write")


if __name__ == '__main__':
    main()
