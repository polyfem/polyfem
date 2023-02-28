#include "SplineParametrizations.hpp"

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	Eigen::VectorXd BSplineParametrization1DTo2D::inverse_eval(const Eigen::VectorXd &y)
	{
		spline_ = std::make_shared<BSplineParametrization2D>(initial_control_points_, knots_, utils::unflatten(y, 2));
		assert(size_ == spline_->vertex_size());
		if (exclude_ends_)
			return utils::flatten(initial_control_points_).segment(2, (initial_control_points_.rows() - 2) * 2);
		else
			return utils::flatten(initial_control_points_);
	}

	Eigen::VectorXd BSplineParametrization1DTo2D::eval(const Eigen::VectorXd &x) const
	{
		Eigen::MatrixXd new_control_points;
		if (exclude_ends_)
		{
			new_control_points = initial_control_points_;
			for (int i = 1; i < new_control_points.rows() - 1; ++i)
				new_control_points.row(i) = x.segment(2 * i - 2, 2);
		}
		else
		{
			new_control_points = utils::unflatten(x, 2);
		}
		Eigen::MatrixXd new_vertices;
		spline_->reparametrize(new_control_points, new_vertices);
		Eigen::VectorXd y = utils::flatten(new_vertices);
		return y;
	}

	Eigen::VectorXd BSplineParametrization1DTo2D::apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd grad;
		spline_->derivative_wrt_params(grad_full, grad);
		if (exclude_ends_)
			return grad.segment(2, (initial_control_points_.rows() - 2) * 2);
		else
			return grad;
	}

	Eigen::VectorXd BSplineParametrization2DTo3D::inverse_eval(const Eigen::VectorXd &y)
	{
		spline_ = std::make_shared<BSplineParametrization3D>(initial_control_point_grid_, knots_u_, knots_v_, y);
		return Eigen::VectorXd();
	}

	Eigen::VectorXd BSplineParametrization2DTo3D::eval(const Eigen::VectorXd &x) const
	{
		return Eigen::VectorXd();
	}

	Eigen::VectorXd BSplineParametrization2DTo3D::apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
	{
		return Eigen::VectorXd();
	}
} // namespace polyfem::solver