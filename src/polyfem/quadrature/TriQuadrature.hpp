#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	class TriQuadrature
	{
	public:
		TriQuadrature();

		void get_quadrature(const int order, Quadrature &quad);
	};
} // namespace polyfem
