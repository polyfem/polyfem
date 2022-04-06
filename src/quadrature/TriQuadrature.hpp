#ifndef TRI_QUADRATURE_HPP
#define TRI_QUADRATURE_HPP

#include <polyfem/Quadrature.hpp>

namespace polyfem
{
	class TriQuadrature
	{
	public:
		TriQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem

#endif //TRI_QUADRATURE_HPP
