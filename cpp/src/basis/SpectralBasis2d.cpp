#include "SpectralBasis2d.hpp"

#include "QuadraticBSpline2d.hpp"
#include "QuadQuadrature.hpp"
#include "MeshNodes.hpp"

#include "LinearSolver.hpp"
#include "FEBasis2d.hpp"
#include "Types.hpp"

#include "Common.hpp"

#include <Eigen/Sparse>

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <map>


namespace poly_fem
{
    using namespace Eigen;

    namespace
    {
        double basis_1d(const double x, const int n)
        {
            if(n == 0)
                return 0.5;
            if(n % 2 == 0)
            {
                return cos(2*M_PI*(n/2)*x);
            }
            else
            {
                return sin(2*M_PI*((n+1)/2)*x);
            }
        }

        double grad_basis_1d(const double x, const int n)
        {
            if(n == 0)
                return 0;
            if(n % 2 == 0)
            {
                return -(2*M_PI*(n/2))*sin(2*M_PI*(n/2)*x);
            }
            else
            {
                return (2*M_PI*((n+1)/2))*cos(2*M_PI*((n+1)/2)*x);
            }
        }

        void basis(const Eigen::MatrixXd &uv, const int n, Eigen::MatrixXd &result)
        {
            const int n_pts = int(uv.rows());
            assert(uv.cols() == 2);

            result.resize(n_pts, 1);

            for(int i = 0; i < n_pts; ++i)
                result(i) = basis_1d(uv(i,0), n) * basis_1d(uv(i,1), n);
        }


        void derivative(const Eigen::MatrixXd &uv, const int n, Eigen::MatrixXd &result)
        {
            const int n_pts = int(uv.rows());
            assert(uv.cols() == 2);

            result.resize(n_pts, 2);

            for(int i = 0; i < n_pts; ++i)
            {
                const double u = uv(i,0);
                const double v = uv(i,1);

                result(i,0) = grad_basis_1d(u, n) * basis_1d(v, n);
                result(i,1) = basis_1d(u, n) * grad_basis_1d(v, n);
            }
        }
    }

    int SpectralBasis2d::build_bases(
        const Mesh2D &mesh,
        const int quadrature_order,
        const int order,
        std::vector< ElementBases > &bases,
        std::vector< ElementBases > &gbases,
        std::vector< LocalBoundary > &local_boundary)
    {
        bases.resize(1);
        ElementBases &b = bases.front();
        b.has_parameterization = false;

        const int n_bases = 2*order + 1;

        b.bases.resize(n_bases);
        b.set_quadrature([quadrature_order](Quadrature &quad){
            QuadQuadrature quad_quadrature;
            quad_quadrature.get_quadrature(quadrature_order, quad);
        });


        for (int j = 0; j < n_bases; ++j) {
            const int global_index = j;

            b.bases[j].init(global_index, j, Eigen::MatrixXd::Zero(1, 2));
            b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { basis(uv, j, val); });
            b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { derivative(uv, j, val); });
        }


        // gbases.resize(1);
        // ElementBases &gb = bases.front();
        // gb.bases.resize(n_bases);
        // gb.set_quadrature([quadrature_order](Quadrature &quad){
        //     QuadQuadrature quad_quadrature;
        //     quad_quadrature.get_quadrature(quadrature_order, quad);
        // });
        // b.has_parameterization = false;

        // for (int j = 0; j < n_bases; ++j) {
        //     const int global_index = j;

        //     gb.bases[j].init(global_index, j, Eigen::Vector2d(0,0));
        //     gb.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { basis(uv, j, val); });
        //     gb.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { derivative(uv, j, val); });
        // }


        return n_bases;
    }

}
