#include "HexBoundarySampler.hpp"


namespace poly_fem {
	bool HexBoundarySampler::sample(const bool is_right_boundary,
		const bool is_bottom_boundary,
		const bool is_left_boundary,
		const bool is_top_boundary,
		const bool is_front_boundary,
		const bool is_back_boundary,
		const int resolution_one_d,
		const bool skip_computation,
		Eigen::MatrixXd &samples)
	{
		const int resolution = resolution_one_d *resolution_one_d;

		int n = 0;
		if(is_right_boundary) n+=resolution;
		if(is_left_boundary) n+=resolution;
		if(is_top_boundary) n+=resolution;
		if(is_bottom_boundary) n+=resolution;
		if(is_front_boundary) n+=resolution;
		if(is_back_boundary) n+=resolution;

		if(n <= 0) return false;

		samples.resize(n, 3);
		if(skip_computation) return true;

		const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(resolution_one_d, 0, 1);

		Eigen::MatrixXd tx(resolution, 1);
		Eigen::MatrixXd ty(resolution, 1);

		for(int i = 0; i < resolution_one_d; ++i)
		{
			for(int j = 0; j < resolution_one_d; ++j)
			{
				tx(i * resolution_one_d + j) = t(i);
				ty(i * resolution_one_d + j) = t(j);
			}
		}


		n = 0;

		if(is_left_boundary){
			samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
			samples.block(n, 1, resolution, 1) = tx;
			samples.block(n, 2, resolution, 1) = ty;

			n += resolution;
		}
		if(is_right_boundary){
			samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
			samples.block(n, 1, resolution, 1) = tx;
			samples.block(n, 2, resolution, 1) = ty;

			n += resolution;
		}



		if(is_bottom_boundary){
			samples.block(n, 0, resolution, 1) = tx;
			samples.block(n, 1, resolution, 1) = ty;
			samples.block(n, 2, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);

			n += resolution;
		}
		if(is_top_boundary){
			samples.block(n, 0, resolution, 1) = tx;
			samples.block(n, 1, resolution, 1) = ty;
			samples.block(n, 2, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);

			n += resolution;
		}


		if(is_front_boundary){
			samples.block(n, 0, resolution, 1) = tx;
			samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
			samples.block(n, 2, resolution, 1) = ty;

			n += resolution;
		}
		if(is_back_boundary){
			samples.block(n, 0, resolution, 1) = tx;
			samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
			samples.block(n, 2, resolution, 1) = ty;

			n += resolution;
		}


		assert(long(n) == samples.rows());

		return true;
	}
}

