#include "QuadBoundarySampler.hpp"


namespace poly_fem {
	bool QuadBoundarySampler::sample(const bool is_right_boundary,
		const bool is_bottom_boundary,
		const bool is_left_boundary,
		const bool is_top_boundary,
		const int resolution,
		const bool skip_computation,
		Eigen::MatrixXd &samples)
	{
		int n = 0;
		if(is_right_boundary) n+=resolution;
		if(is_left_boundary) n+=resolution;
		if(is_top_boundary) n+=resolution;
		if(is_bottom_boundary) n+=resolution;

		if(n <= 0) return false;

		samples.resize(n, 2);
		if(skip_computation) return true;

		const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(resolution, 0, 1);
		samples.setConstant(-1);

		n = 0;
		if(is_right_boundary){
			samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
			samples.block(n, 1, resolution, 1) = t;

			n += resolution;
		}

		if(is_top_boundary){
			samples.block(n, 0, resolution, 1) = t;
			samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);

			n += resolution;
		}

		if(is_left_boundary){
			samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
			samples.block(n, 1, resolution, 1) = t;

			n += resolution;
		}

		if(is_bottom_boundary){
			samples.block(n, 0, resolution, 1) = t;
			samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);

			n += resolution;
		}

		assert(long(n) == samples.rows());
		assert(samples.minCoeff()  >= 0);
		assert(samples.maxCoeff()  <= 1);

		return true;
	}
}
