#pragma once

#include "Parametrization.hpp"

#include <Eigen/Core>

namespace polyfem
{
	class State;
} // namespace polyfem

namespace polyfem::solver
{
	class BoundedBiharmonicWeights2Dto3D : public Parametrization
	{
	public:
		BoundedBiharmonicWeights2Dto3D(const int num_control_vertices, const int num_vertices, const Eigen::MatrixXd &V_surface, const Eigen::MatrixXi &F_surface) : num_control_vertices_(num_control_vertices), num_vertices_(num_vertices), V_surface_(V_surface), F_surface_(F_surface), allow_rotations_(true) {}
		BoundedBiharmonicWeights2Dto3D(const int num_control_vertices, const int num_vertices, const State &state, const bool allow_rotations);

		// Should only be called to initialize the parameter, when the shape matches the initial control points.
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		int size(const int x_size) const override { return num_vertices_ * 3; }
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const override;

		Eigen::MatrixXd get_bbw_weights() { return bbw_weights_; }

	private:
		void compute_faces_for_partial_vertices(const Eigen::MatrixXd &V, Eigen::MatrixXi &F) const;

		int optimal_new_control_point_idx(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &boundary_loop, const std::vector<int> &existing_points) const;

		const int num_control_vertices_;
		const int num_vertices_;
		Eigen::MatrixXd control_points_;
		Eigen::MatrixXd bbw_weights_;
		Eigen::MatrixXd boundary_bbw_weights_;

		Eigen::MatrixXd V_surface_;
		Eigen::MatrixXi F_surface_;

		Eigen::VectorXd y_start;

		bool invoked_inverse_eval_ = false;

		const bool allow_rotations_;
	};

} // namespace polyfem::solver