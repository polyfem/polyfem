#ifndef HEX_BOUNDARY_SAMPLER_HPP
#define HEX_BOUNDARY_SAMPLER_HPP

#include <Eigen/Dense>

namespace poly_fem {

	class HexBoundarySampler {
	public:
		static bool sample(const bool is_right_boundary,
		const bool is_bottom_boundary,
		const bool is_left_boundary,
		const bool is_top_boundary,
		const bool is_front_boundary,
		const bool is_back_boundary,
		const int resolution_one_d,
		const bool skip_computation,
		Eigen::MatrixXd &samples);
	};
}

#endif //HEX_BOUNDARY_SAMPLER_HPP
