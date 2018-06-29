#ifndef TRI_QUADRATURE_HPP
#define TRI_QUADRATURE_HPP

#include <polyfem/Quadrature.hpp>

namespace poly_fem
{
    class TriQuadrature
    {
    public:
        TriQuadrature();

        void get_quadrature(const int order, Quadrature &quad);
    };
}

#endif //TRI_QUADRATURE_HPP
