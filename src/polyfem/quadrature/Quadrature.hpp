#pragma once

#include <Eigen/Dense>

namespace polyfem::quadrature
{
	class Quadrature
	{
	public:
		Eigen::MatrixXd points;
		Eigen::VectorXd weights;
	};
} // namespace polyfem::quadrature
