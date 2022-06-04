#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	class QuadQuadrature
	{
	public:
		QuadQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem
