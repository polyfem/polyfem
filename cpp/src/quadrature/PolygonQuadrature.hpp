#ifndef POLYGON_QUADRATURE_HPP
#define POLYGON_QUADRATURE_HPP

#include "Quadrature.hpp"

namespace poly_fem
{
    class PolygonQuadrature
    {
    public:
        PolygonQuadrature();

        void get_quadrature(const Eigen::MatrixXd &poly, const int order, Quadrature &quad);
    };
}

#endif //POLYGON_QUADRATURE_HPP
