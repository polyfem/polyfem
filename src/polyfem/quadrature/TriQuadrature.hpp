#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	namespace quadrature
	{
		class TriQuadrature
		{
		public:
			TriQuadrature(bool use_corner_quadrature = false);

			void get_quadrature(const int order, Quadrature &quad);

		private:
			bool use_corner_quadrature_;
		};
	} // namespace quadrature
} // namespace polyfem
