#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	class TetQuadrature
	{
	public:
		TetQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem
