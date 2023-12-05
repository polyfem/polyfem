#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	namespace quadrature
	{
		class PrismQuadrature
		{
		public:
			PrismQuadrature();

			void get_quadrature(const int order, const int order_h, Quadrature &quad);
		};
	} // namespace quadrature
} // namespace polyfem
