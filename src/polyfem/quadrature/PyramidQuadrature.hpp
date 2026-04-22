#pragma once

#include "Quadrature.hpp"

namespace polyfem
{
	namespace quadrature
	{
		class PyramidQuadrature
		{
		public:
			PyramidQuadrature();

			void get_quadrature(const int order, Quadrature &quad);

		private:
			bool use_corner_quadrature_;
		};
	} // namespace quadrature
} // namespace polyfem
