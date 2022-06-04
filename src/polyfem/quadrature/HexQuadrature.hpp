#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	class HexQuadrature
	{
	public:
		HexQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem
