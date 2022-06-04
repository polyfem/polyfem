#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	class LineQuadrature
	{
	public:
		LineQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem
