#include <iostream>
#include <cassert>
#include <cmath>
#include "EigenAdapters.hpp"
#include <algorithms/solve.hpp>
#include <algorithms/minimize.hpp>

using namespace miso;

// Unit right triangle / tetrahedron node coordinates from polyfem's
// autogen::p_nodes_2d / p_nodes_3d, in the exact order extract_nodes() produces.

// Triangle, order 1 (3 nodes)
static const double tri1[][2] = {
    {0, 0}, {1, 0}, {0, 1}
};

// Triangle, order 2 (6 nodes)
static const double tri2[][2] = {
    {0, 0}, {1, 0}, {0, 1},
    {0.5, 0}, {0.5, 0.5}, {0, 0.5}
};

// Triangle, order 3 (10 nodes)
static const double tri3[][2] = {
    {0, 0}, {1, 0}, {0, 1},
    {1./3, 0}, {2./3, 0},
    {2./3, 1./3}, {1./3, 2./3},
    {0, 2./3}, {0, 1./3},
    {1./3, 1./3}
};

// Triangle, order 4 (15 nodes)
static const double tri4[][2] = {
    {0, 0}, {1, 0}, {0, 1},
    {0.25, 0}, {0.5, 0}, {0.75, 0},
    {0.75, 0.25}, {0.5, 0.5}, {0.25, 0.75},
    {0, 0.75}, {0, 0.5}, {0, 0.25},
    {0.25, 0.25}, {0.25, 0.5}, {0.5, 0.25}
};

// Tetrahedron, order 1 (4 nodes)
static const double tet1[][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
};

// Tetrahedron, order 2 (10 nodes)
static const double tet2[][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    {0.5, 0, 0}, {0.5, 0.5, 0}, {0, 0.5, 0},
    {0, 0, 0.5}, {0.5, 0, 0.5}, {0, 0.5, 0.5}
};

// Tetrahedron, order 3 (20 nodes)
static const double tet3[][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    {1./3, 0, 0}, {2./3, 0, 0},
    {2./3, 1./3, 0}, {1./3, 2./3, 0},
    {0, 2./3, 0}, {0, 1./3, 0},
    {0, 0, 1./3}, {0, 0, 2./3},
    {2./3, 0, 1./3}, {1./3, 0, 2./3},
    {0, 2./3, 1./3}, {0, 1./3, 2./3},
    {1./3, 1./3, 0}, {1./3, 0, 1./3},
    {1./3, 1./3, 1./3}, {0, 1./3, 1./3}
};

template<int N, int D>
Eigen::MatrixXd make_cp(const double (&coords)[N][D])
{
    Eigen::MatrixXd cp(N, D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            cp(i, j) = coords[i][j];
    return cp;
}

template<int N, int D>
Eigen::MatrixXd make_translated(const double (&coords)[N][D], double dx, double dy, double dz = 0)
{
    Eigen::MatrixXd cp(N, D);
    double d[3] = {dx, dy, dz};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            cp(i, j) = coords[i][j] + d[j];
    return cp;
}

int main()
{
    int passed = 0, failed = 0;
    auto check = [&](const char *name, bool condition) {
        if (condition) {
            std::cout << "  PASS: " << name << "\n";
            ++passed;
        } else {
            std::cout << "  FAIL: " << name << "\n";
            ++failed;
        }
    };

    // ==================== Static validity (Val) ====================
    std::cout << "=== Static validity ===\n";
    Parameters solve_params;
    solve_params.constraintEpsilon = {0.0};
    solve_params.findOne = true;
    {
        // P1 Triangle: JacEval should return positive determinant
        auto cp = make_cp(tri1);
        auto det = JacEval_P1Tri(rv<3>(cp, 0, 0, tri1_perm), rv<3>(cp, 0, 1, tri1_perm));
        check("P1 Tri static (det > 0)", lower(det) > 0);
    }
    {
        // P2 Triangle
        auto cp = make_cp(tri2);
        auto sols = solve(make_p2tri_val(cp, 0), solve_params);
        check("P2 Tri static (valid)", sols.empty());
    }
    {
        // P3 Triangle
        auto cp = make_cp(tri3);
        auto sols = solve(make_p3tri_val(cp, 0), solve_params);
        check("P3 Tri static (valid)", sols.empty());
    }
    {
        // P4 Triangle
        auto cp = make_cp(tri4);
        auto sols = solve(make_p4tri_val(cp, 0), solve_params);
        check("P4 Tri static (valid)", sols.empty());
    }
    {
        // P1 Tetrahedron
        auto cp = make_cp(tet1);
        auto det = JacEval_P1Tet(rv<4>(cp, 0, 0, tet1_perm), rv<4>(cp, 0, 1, tet1_perm), rv<4>(cp, 0, 2, tet1_perm));
        check("P1 Tet static (det > 0)", lower(det) > 0);
    }
    {
        // P2 Tetrahedron
        auto cp = make_cp(tet2);
        auto sols = solve(make_p2tet_val(cp, 0), solve_params);
        check("P2 Tet static (valid)", sols.empty());
    }
    {
        // P3 Tetrahedron
        auto cp = make_cp(tet3);
        auto sols = solve(make_p3tet_val(cp, 0), solve_params);
        check("P3 Tet static (valid)", sols.empty());
    }

    // ==================== Continuous validity (CGV) ====================
    // Apply a uniform translation (1, 1) for 2D, (1, 1, 1) for 3D.
    // A pure translation preserves validity, so minimize should return t_lo >= 1.
    std::cout << "\n=== Continuous validity (uniform translation) ===\n";
    Parameters minimize_params;
    minimize_params.targetPrecision = 1e-6;
    minimize_params.constraintEpsilon = {0.0};
    {
        auto cp1 = make_cp(tri1);
        auto cp2 = make_translated(tri1, 1.0, 1.0);
        auto result = minimize(make_p1tri_cgv(cp1, cp2, 0), minimize_params);
        check("P1 Tri CGV (t_lo >= 1)", lower(result) >= 1.0);
    }
    {
        auto cp1 = make_cp(tri2);
        auto cp2 = make_translated(tri2, 1.0, 1.0);
        auto result = minimize(make_p2tri_cgv(cp1, cp2, 0), minimize_params);
        check("P2 Tri CGV (t_lo >= 1)", lower(result) >= 1.0);
    }
    {
        auto cp1 = make_cp(tri3);
        auto cp2 = make_translated(tri3, 1.0, 1.0);
        auto result = minimize(make_p3tri_cgv(cp1, cp2, 0), minimize_params);
        check("P3 Tri CGV (t_lo >= 1)", lower(result) >= 1.0);
    }
    {
        auto cp1 = make_cp(tri4);
        auto cp2 = make_translated(tri4, 1.0, 1.0);
        auto result = minimize(make_p4tri_cgv(cp1, cp2, 0), minimize_params);
        check("P4 Tri CGV (t_lo >= 1)", lower(result) >= 1.0);
    }
    {
        auto cp1 = make_cp(tet1);
        auto cp2 = make_translated(tet1, 1.0, 1.0, 1.0);
        auto result = minimize(make_p1tet_cgv(cp1, cp2, 0), minimize_params);
        check("P1 Tet CGV (t_lo >= 1)", lower(result) >= 1.0);
    }
    {
        auto cp1 = make_cp(tet2);
        auto cp2 = make_translated(tet2, 1.0, 1.0, 1.0);
        auto result = minimize(make_p2tet_cgv(cp1, cp2, 0), minimize_params);
        check("P2 Tet CGV (t_lo >= 1)", lower(result) >= 1.0);
    }
    {
        auto cp1 = make_cp(tet3);
        auto cp2 = make_translated(tet3, 1.0, 1.0, 1.0);
        auto result = minimize(make_p3tet_cgv(cp1, cp2, 0), minimize_params);
        check("P3 Tet CGV (t_lo >= 1)", lower(result) >= 1.0);
    }

    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}
