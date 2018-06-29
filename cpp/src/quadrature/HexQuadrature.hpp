#ifndef HEX_QUADRATURE_HPP
#define HEX_QUADRATURE_HPP

#include <polyfem/Quadrature.hpp>

namespace poly_fem
{
    class HexQuadrature
    {
    public:
        HexQuadrature();

        void get_quadrature(const int order, Quadrature &quad);
    };
}

#endif //HEX_QUADRATURE_HPP