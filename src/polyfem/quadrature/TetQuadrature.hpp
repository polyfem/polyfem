#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	namespace quadrature
	{
		class TetQuadrature
		{
		public:
			TetQuadrature();

			void get_quadrature(const int order, Quadrature &quad);
		};
	} // namespace quadrature
} // namespace polyfem
