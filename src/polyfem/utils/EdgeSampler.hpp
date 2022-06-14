#pragma once

#include <Eigen/Dense>

namespace polyfem
{
	namespace utils
	{
		class EdgeSampler
		{
		public:
			static void sample_2d_simplex(const int resolution, Eigen::MatrixXd &samples);
			static void sample_2d_cube(const int resolution, Eigen::MatrixXd &samples);

			static void sample_3d_simplex(const int resolution, Eigen::MatrixXd &samples);
			static void sample_3d_cube(const int resolution, Eigen::MatrixXd &samples);
		};
	} // namespace utils
} // namespace polyfem
