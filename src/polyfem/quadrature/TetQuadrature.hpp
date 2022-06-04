#ifndef TET_QUADRATURE_HPP
#define TET_QUADRATURE_HPP

#include <polyfem/Quadrature.hpp>

namespace polyfem
{
	class TetQuadrature
	{
	public:
		TetQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem

#endif //TET_QUADRATURE_HPP
