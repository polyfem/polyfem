#ifndef QUAD_QUADRATURE_HPP
#define QUAD_QUADRATURE_HPP

#include "Quadrature.hpp"

namespace poly_fem
{
    class QuadQuadrature
    {
    public:
        QuadQuadrature();

        void get_quadrature(const int order, Quadrature &quad);
    };
}

#endif //QUAD_QUADRATURE_HPP