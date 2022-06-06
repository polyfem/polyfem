#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	namespace quadrature
	{
		class LineQuadrature
		{
		public:
			LineQuadrature();

			void get_quadrature(const int order, Quadrature &quad);
		};
	} // namespace quadrature
} // namespace polyfem
