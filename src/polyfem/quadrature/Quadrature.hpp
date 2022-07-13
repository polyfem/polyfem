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

			int size() const
			{
				assert(points.size() == weights.size());
				return points.size();
			}
		};
	} // namespace quadrature
} // namespace polyfem
