#ifndef LINE_QUADRATURE_HPP
#define LINE_QUADRATURE_HPP

#include <polyfem/Quadrature.hpp>

namespace polyfem
{
	class LineQuadrature
	{
	public:
		LineQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem

#endif //LINE_QUADRATURE_HPP