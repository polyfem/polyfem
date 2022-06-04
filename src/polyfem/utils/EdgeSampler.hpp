#ifndef EDGE_SAMPLER_HPP
#define EDGE_SAMPLER_HPP

#include <Eigen/Dense>

namespace polyfem
{

	class EdgeSampler
	{
	public:
		static void sample_2d_simplex(const int resolution, Eigen::MatrixXd &samples);
		static void sample_2d_cube(const int resolution, Eigen::MatrixXd &samples);

		static void sample_3d_simplex(const int resolution, Eigen::MatrixXd &samples);
		static void sample_3d_cube(const int resolution, Eigen::MatrixXd &samples);
	};
} // namespace polyfem

#endif //EDGE_SAMPLER_HPP
