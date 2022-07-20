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
				assert(points.rows() == weights.size());
				return points.rows();
			}
		};
	} // namespace quadrature
} // namespace polyfem
