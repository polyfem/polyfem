#include <polyfem/TetQuadrature.hpp>
#include <polyfem/LineQuadrature.hpp>

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

namespace polyfem
{
    namespace
    {
        void get_weight_and_points(const int order, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
        {
            switch(order)
            {
				#include <polyfem/auto_tetrahedron.ipp>

                default:
                assert(false);
            };
        }
    }

    TetQuadrature::TetQuadrature()
    { }

    void TetQuadrature::get_quadrature(const int order, Quadrature &quad)
    {
        Quadrature tmp;

        get_weight_and_points(order, quad.points, quad.weights);

        assert(fabs(quad.weights.sum()-1)<1e-12);
        assert(quad.points.minCoeff()>=0 && quad.points.maxCoeff()<=1);

        assert(quad.points.rows()==quad.weights.size());

        quad.weights/=6;
    }
}
