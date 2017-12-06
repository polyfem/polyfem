#ifndef LINE_QUADRATURE_HPP
#define LINE_QUADRATURE_HPP

#include "Quadrature.hpp"

namespace poly_fem
{
    class LineQuadrature
    {
    public:
        LineQuadrature();

        void get_quadrature(const int order, Quadrature &quad);
    };
}

#endif //LINE_QUADRATURE_HPP