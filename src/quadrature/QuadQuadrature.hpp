#ifndef QUAD_QUADRATURE_HPP
#define QUAD_QUADRATURE_HPP

#include <polyfem/Quadrature.hpp>

namespace polyfem
{
	class QuadQuadrature
	{
	public:
		QuadQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem

#endif //QUAD_QUADRATURE_HPP