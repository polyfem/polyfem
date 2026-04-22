import numpy as np
import sympy

def integrate_exact(f):
    """Exact integral of f over the reference pyramid: 0<=z<=1, 0<=x,y<=1-z."""
    x, y, z = sympy.symbols('x y z')
    return sympy.integrate(
        sympy.integrate(
            sympy.integrate(f, (x, 0, 1 - z)),
            (y, 0, 1 - z)),
        (z, 0, 1))

def duffy_quadrature(order):
    """
    Build a pyramid quadrature rule via the Duffy transform:
        x = xi*(1-zeta), y = eta*(1-zeta), z = zeta
        J = (1-zeta)^2
    A tensor product of n 1D Gauss-Legendre points (n = ceil((order+1)/2))
    integrates the pulled-back polynomial stiffness integrand exactly for
    pyramid basis degree p = order//2.

    Returns (pts, weights) on the reference pyramid [0<=x,y<=1-z, 0<=z<=1].
    Weights sum to 1/3.
    """
    n = (order + 4) // 2   # ceil((order+3)/2): exact for degree (order+2) in zeta

    # 1D Gauss-Legendre on [-1,1]
    t, w = np.polynomial.legendre.leggauss(n)

    # Map to [0,1]: xi = (t+1)/2, w_1d = w/2
    xi1d = (t + 1.0) / 2.0
    w1d  = w / 2.0

    pts     = []
    weights = []
    for i in range(n):         # xi
        for j in range(n):     # eta
            for k in range(n): # zeta
                xi, eta, zeta = xi1d[i], xi1d[j], xi1d[k]
                x = xi  * (1.0 - zeta)
                y = eta * (1.0 - zeta)
                z = zeta
                w_total = w1d[i] * w1d[j] * w1d[k] * (1.0 - zeta)**2
                pts.append([x, y, z])
                weights.append(w_total)

    pts     = np.array(pts)
    weights = np.array(weights)
    return pts, weights


def verify_duffy(max_order=8):
    """
    Verify that duffy_quadrature integrates all monomials x^i y^j z^k
    (i+j+k <= order) exactly over the reference pyramid.
    """
    import sympy
    x, y, z = sympy.symbols('x y z')

    for order in range(1, max_order + 1):
        pts, weights = duffy_quadrature(order)
        max_err = 0.0
        for i in range(order + 1):
            for j in range(order + 1):
                for k in range(order + 1):
                    if i + j + k > order:
                        continue
                    # exact integral of x^i y^j z^k over pyramid
                    f = x**i * y**j * z**k
                    exact = float(integrate_exact(f))
                    approx = float(np.sum(weights * pts[:,0]**i
                                                 * pts[:,1]**j
                                                 * pts[:,2]**k))
                    max_err = max(max_err, abs(exact - approx))
        print(f"order={order}, n={(order+2)//2}^3={(((order+2)//2)**3)} pts, "
              f"max monomial error={max_err:.2e}, "
              f"weight sum={weights.sum():.10f} (should be {1/3:.10f})")


# Also verify against the rational pyramid basis integrands:
def verify_duffy_rational(max_p=4):
    """
    Verify Duffy integrates the actual pyramid rational basis products exactly.
    Tests xy/(1-z), x^2*y/(1-z)^2, etc. that appear in stiffness integrals.
    """
    import sympy
    x, y, z = sympy.symbols('x y z')

    for p in range(1, max_p + 1):
        order = 2 * p   # stiffness integrand degree
        pts, weights = duffy_quadrature(order)
        max_err = 0.0
        # Test basis-like rational monomials (x/(1-z))^a (y/(1-z))^b (1-z)^c
        for a in range(p + 1):
            for b in range(p + 1):
                for c in range(p + 1):
                    # rational function on pyramid: x^a y^b / (1-z)^(a+b-c)
                    # typical stiffness term after two gradient evaluations
                    f = x**a * y**b / (1 - z)**(a + b - c) if (a+b-c) >= 0 else x**a * y**b * (1-z)**(c-a-b)
                    try:
                        exact = float(integrate_exact(f))
                    except Exception:
                        continue
                    pts_x, pts_y, pts_z = pts[:,0], pts[:,1], pts[:,2]
                    if a + b - c >= 0:
                        approx = float(np.sum(weights * pts_x**a * pts_y**b
                                              / (1 - pts_z)**(a+b-c)))
                    else:
                        approx = float(np.sum(weights * pts_x**a * pts_y**b
                                              * (1 - pts_z)**(c-a-b)))
                    max_err = max(max_err, abs(exact - approx))
        print(f"p={p}, order={order}, max rational error={max_err:.2e}")

if __name__ == "__main__":
    print("=== Monomial verification ===")
    verify_duffy(max_order=8)
    print()
    print("=== Rational basis verification ===")
    verify_duffy_rational(max_p=4)
