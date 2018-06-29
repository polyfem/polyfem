#ifndef TET_QUADRATURE_HPP
#define TET_QUADRATURE_HPP

#include <polyfem/Quadrature.hpp>

namespace poly_fem
{
    class TetQuadrature
    {
    public:
        TetQuadrature();

        void get_quadrature(const int order, Quadrature &quad);
    };
}

#endif //TET_QUADRATURE_HPP
