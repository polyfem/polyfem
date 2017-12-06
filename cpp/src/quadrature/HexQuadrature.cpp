#include "HexQuadrature.hpp"
#include "LineQuadrature.hpp"

#include <vector>
#include <cassert>
#include <cmath>

namespace poly_fem
{

    HexQuadrature::HexQuadrature()
    { }

    void HexQuadrature::get_quadrature(const int order, Quadrature &quad)
    {
        Quadrature tmp;
        LineQuadrature one_d_quad;
        one_d_quad.get_quadrature(order, tmp);

        const long n_quad_pts = tmp.weights.size();

        quad.points = Eigen::MatrixXd(n_quad_pts*n_quad_pts*n_quad_pts, 3);
        quad.weights = Eigen::MatrixXd(n_quad_pts*n_quad_pts*n_quad_pts, 1);


        for (long i = 0; i < n_quad_pts; ++i)
        {
            for (long j = 0; j < n_quad_pts; ++j)
            {
                for (long k = 0; k < n_quad_pts; ++k)
                {
                    quad.points.row((i*n_quad_pts+j)*n_quad_pts+k) = Eigen::Vector3d(tmp.points(k), tmp.points(j), tmp.points(i));
                    quad.weights((i*n_quad_pts+j)*n_quad_pts+k) = tmp.weights(i)*tmp.weights(j)*tmp.weights(i);
                }
            }
        }
    }
}