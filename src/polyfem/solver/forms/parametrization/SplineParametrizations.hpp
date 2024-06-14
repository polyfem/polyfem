#pragma once

#include "Parametrization.hpp"

#include <Eigen/Core>

namespace polyfem
{
	class State;
	class BSplineParametrization2D;
	class BSplineParametrization3D;
} // namespace polyfem

namespace polyfem::solver
{
	class BSplineParametrization1DTo2D : public Parametrization
	{
	public:
		BSplineParametrization1DTo2D(const Eigen::MatrixXd &initial_control_points, const Eigen::VectorXd &knots, const int num_vertices, const bool exclude_ends = true)
			: initial_control_points_(initial_control_points), knots_(knots), size_(num_vertices), exclude_ends_(exclude_ends) {}

		// Should only be called to initialize the parameter, when the shape matches the initial control points.
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		inline int size(const int x_size) const override
		{
			return 2 * size_;
		};
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const override;

	private:
		const Eigen::MatrixXd initial_control_points_;
		const Eigen::VectorXd knots_;

		bool invoked_inverse_eval_ = false;

		const int size_;

		std::shared_ptr<BSplineParametrization2D> spline_;

		const bool exclude_ends_;
	};

	class BSplineParametrization2DTo3D : public Parametrization
	{
	public:
		BSplineParametrization2DTo3D(const Eigen::MatrixXd &initial_control_point_grid, const Eigen::VectorXd &knots_u, const Eigen::VectorXd &knots_v, const bool exclude_ends = true) : initial_control_point_grid_(initial_control_point_grid), knots_u_(knots_u), knots_v_(knots_v), exclude_ends_(exclude_ends) {}

		// Should only be called to initialize the parameter, when the shape matches the initial control points.
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		int size(const int x_size) const override { return 3 * u_.size() * v_.size(); }
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const override;

	private:
		const Eigen::MatrixXd initial_control_point_grid_;
		const Eigen::VectorXd knots_u_;
		const Eigen::VectorXd knots_v_;

		std::shared_ptr<BSplineParametrization3D> spline_;

		Eigen::VectorXd u_;
		Eigen::VectorXd v_;

		bool invoked_inverse_eval_ = false;

		const bool exclude_ends_;
	};

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