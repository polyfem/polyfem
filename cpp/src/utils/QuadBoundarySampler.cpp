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

		n = 0;
		if(is_right_boundary){
			samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
			samples.block(n, 1, resolution, 1) = t;

			n += resolution;
		}

		if(is_top_boundary){
			samples.block(n, 0, resolution, 1) = t;
			samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);

			n += resolution;
		}

		if(is_left_boundary){
			samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
			samples.block(n, 1, resolution, 1) = t;

			n += resolution;
		}

		if(is_bottom_boundary){
			samples.block(n, 0, resolution, 1) = t;
			samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);

			n += resolution;
		}

		assert(long(n) == samples.rows());

		return true;
	}
}



















// const int resolution = resolution_one_d *resolution_one_d;

				// const int n_x = mesh.n_x;
				// const int n_y = mesh.n_y;
				// const int n_z = mesh.n_z;

				// const bool has_left = (el(0) % ((n_x + 1)*(n_y + 1))) % (n_x + 1) != 0;
				// const bool has_right = (el(2) % ((n_x + 1)*(n_y + 1))) % (n_x + 1) != n_x;

				// const bool has_top = (el(0) % ((n_x + 1)*(n_y + 1))) / (n_x + 1) != 0;
				// const bool has_bottom = (el(2) % ((n_x + 1)*(n_y + 1))) / (n_x + 1) != n_y;

				// const bool has_front = el(4) < (n_x + 1) * (n_y + 1) * n_z;
				// const bool has_back = el(0) >= (n_x + 1) * (n_y + 1);

				// int n = 0;
				// if(!has_left) n+=resolution;
				// if(!has_right) n+=resolution;
				// if(!has_bottom) n+=resolution;
				// if(!has_top) n+=resolution;
				// if(!has_front) n+=resolution;
				// if(!has_back) n+=resolution;

				// if(n <= 0) return false;

				// const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(resolution_one_d, 0, 1);

				// Eigen::MatrixXd tx(resolution, 1);
				// Eigen::MatrixXd ty(resolution, 1);

				// for(int i = 0; i < resolution_one_d; ++i)
				// {
				// 	for(int j = 0; j < resolution_one_d; ++j)
				// 	{
				// 		tx(i * resolution_one_d + j) = t(i);
				// 		ty(i * resolution_one_d + j) = t(j);
				// 	}
				// }

				// samples.resize(n, 3);
				// n = 0;

				// if(!has_left){
				// 	samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
				// 	samples.block(n, 1, resolution, 1) = tx;
				// 	samples.block(n, 2, resolution, 1) = ty;

				// 	n += resolution;
				// }
				// if(!has_right){
				// 	samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
				// 	samples.block(n, 1, resolution, 1) = tx;
				// 	samples.block(n, 2, resolution, 1) = ty;

				// 	n += resolution;
				// }



				// if(!has_bottom){
				// 	samples.block(n, 0, resolution, 1) = tx;
				// 	samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
				// 	samples.block(n, 2, resolution, 1) = ty;

				// 	n += resolution;
				// }
				// if(!has_top){
				// 	samples.block(n, 0, resolution, 1) = tx;
				// 	samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
				// 	samples.block(n, 2, resolution, 1) = ty;

				// 	n += resolution;
				// }


				// if(!has_front){
				// 	samples.block(n, 0, resolution, 1) = tx;
				// 	samples.block(n, 1, resolution, 1) = ty;
				// 	samples.block(n, 2, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);

				// 	n += resolution;
				// }
				// if(!has_back){
				// 	samples.block(n, 0, resolution, 1) = tx;
				// 	samples.block(n, 1, resolution, 1) = ty;
				// 	samples.block(n, 2, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);

				// 	n += resolution;
				// }



				// // std::cout<<samples<<std::endl;
				// // igl::viewer::Viewer &viewer = State::state().viewer;
				// // viewer.data.add_points(samples, Eigen::MatrixXd::Zero(samples.rows(), 3));
				// // viewer.launch();

				// assert(long(n) == samples.rows());