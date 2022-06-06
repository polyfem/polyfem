#pragma once

#include <Eigen/Dense>

namespace polyfem
{
	namespace quadrature
	{
		class Quadrature
		{
		public:
			Eigen::MatrixXd points;
			Eigen::VectorXd weights;
		};
	} // namespace quadrature
} // namespace polyfem
