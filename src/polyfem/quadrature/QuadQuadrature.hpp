#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	namespace quadrature
	{
		class QuadQuadrature
		{
		public:
			QuadQuadrature();

			void get_quadrature(const int order, Quadrature &quad);
		};
	} // namespace quadrature
} // namespace polyfem
