#pragma once

#include "Parametrization.hpp"

#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/utils/BSplineParametrization.hpp>

#include <Eigen/Core>
#include <map>

namespace polyfem::solver
{
	class BSplineParametrization1DTo2D : public Parametrization
	{
	public:
		BSplineParametrization1DTo2D(const Eigen::MatrixXd &initial_control_points, const Eigen::VectorXd &knots, const bool exclude_ends = true) : initial_control_points_(initial_control_points), knots_(knots), exclude_ends_(exclude_ends) {}

		// Should only be called to initialize the parameter, when the shape matches the initial control points.
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		int size(const int x_size) const override { return t_.size(); }
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const override;

	private:
		const Eigen::MatrixXd initial_control_points_;
		const Eigen::VectorXd knots_;

		Eigen::VectorXd t_;

		std::shared_ptr<BSplineParametrization2D> spline_;

		const bool exclude_ends_;
	};

	class BSplineParametrization2DTo3D : public Parametrization
	{
	public:
		BSplineParametrization2DTo3D(const Eigen::MatrixXd &initial_control_point_grid, const Eigen::VectorXd &knots_u, const Eigen::VectorXd &knots_v, const bool exclude_ends = true) : initial_control_point_grid_(initial_control_point_grid), knots_u_(knots_u), knots_v_(knots_v), exclude_ends_(exclude_ends) {}

		// Should only be called to initialize the parameter, when the shape matches the initial control points.
		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override;
		int size(const int x_size) const override { return u_.size() * v_.size(); }
		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const override;

	private:
		const Eigen::MatrixXd initial_control_point_grid_;
		const Eigen::VectorXd knots_u_;
		const Eigen::VectorXd knots_v_;

		std::shared_ptr<BSplineParametrization3D> spline_;

		Eigen::VectorXd u_;
		Eigen::VectorXd v_;

		const bool exclude_ends_;
	};
} // namespace polyfem::solver