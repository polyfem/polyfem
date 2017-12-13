#ifndef QUAD_BOUNDARY_SAMPLER_HPP
#define QUAD_BOUNDARY_SAMPLER_HPP

#include <Eigen/Dense>

namespace poly_fem {

	class QuadBoundarySampler {
	public:
		static bool sample(const bool is_right_boundary,
		const bool is_bottom_boundary,
		const bool is_left_boundary,
		const bool is_top_boundary,
		const int resolution,
		const bool skip_computation,
		Eigen::MatrixXd &samples);
	};
}

#endif //QUAD_BOUNDARY_SAMPLER_HPP
