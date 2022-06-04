#pragma once

#include <Eigen/Dense>

namespace polyfem
{
	class Quadrature
	{
	public:
		Eigen::MatrixXd points;
		Eigen::VectorXd weights;
	};
} // namespace polyfem
