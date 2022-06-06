#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	namespace quadrature
	{
		class HexQuadrature
		{
		public:
			HexQuadrature();

			void get_quadrature(const int order, Quadrature &quad);
		};
	} // namespace quadrature
} // namespace polyfem
