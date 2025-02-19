#include "BSplineParametrization.hpp"
#include <polyfem/utils/Logger.hpp>

namespace polyfem
{
	// void BSplineParametrization::get_parameters(const Eigen::MatrixXd &V, Eigen::MatrixXd &control_points)
	// {
	// 	bool mesh_changed = V.rows() != num_vertices;
	// 	get_parameters(V, control_points, mesh_changed);
	// }

	BSplineParametrization2D::BSplineParametrization2D(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &knots, const Eigen::MatrixXd &V) : BSplineParametrization(V), dim(control_points.cols())
	{
		assert(dim == 2);
		// Deduce the t parameter of all of the points in the spline sections
		double tol = 1e-4;
		curve.set_control_points(control_points);
		curve.set_knots(knots);
		std::vector<int> unused;
		for (int i = 0; i < V.rows(); ++i)
			node_ids_.push_back(i);
		for (const auto &b : node_ids_)
		{
			Eigen::MatrixXd point = V.block(b, 0, 1, dim);
			auto t = curve.approximate_inverse_evaluate(point);
			double distance = (point - curve.evaluate(t)).norm();

			if (distance > tol)
			{
				logger().error("Could not find a valid t for deducing spline parametrization. Distance: {}, point: {}, {}", distance, point(0), point(1));
				unused.push_back(b);
				continue;
			}

			node_id_to_t_[b] = t;
		}

		if (unused.size() > 0)
			log_and_throw_error("Some nodes do not take part in the spline parametrization!");

		// Remove nodes that do not have a parametrization.
		// for (const auto &i : unused)
		// {
		// 	auto loc = std::find(node_ids_.begin(), node_ids_.end(), i);
		// 	if (loc == node_ids_.end())
		// 		logger().error("Error removing unused node.");
		// 	node_ids_.erase(loc);
		// }
		logger().info("Number of useful boundary nodes in spline parametrization: {}", node_ids_.size());
	}

	void BSplineParametrization2D::reparametrize(const Eigen::MatrixXd &control_points, Eigen::MatrixXd &newV)
	{
		// Given new control parameters and the t parameter precomputed, compute new V
		curve.set_control_points(control_points);
		newV.setZero(node_ids_.size(), 2);
		for (const auto &b : node_ids_)
		{
			auto new_val = curve.evaluate(node_id_to_t_.at(b));
			newV.block(b, 0, 1, dim) = new_val;
		}
	}

	void BSplineParametrization2D::get_parameters(const Eigen::MatrixXd &V, Eigen::MatrixXd &control_points, const bool mesh_changed)
	{
		assert(!mesh_changed);
		Eigen::MatrixXd boundary_points(node_ids_.size(), dim), t_params(node_ids_.size(), 1);
		bool boundary_changed = false;
		for (int i = 0; i < node_ids_.size(); ++i)
		{
			auto old_point = curve.evaluate(node_id_to_t_.at(node_ids_[i]));
			auto new_point = V.block(node_ids_[i], 0, 1, dim);
			boundary_points.block(i, 0, 1, dim) = new_point;
			t_params(i) = node_id_to_t_.at(node_ids_[i]);
			double difference = (old_point - new_point).norm();
			if (difference > 1e-4)
			{
				boundary_changed = true;
			}
		}
		if (boundary_changed)
		{
			// Deduce parameter values from vertex positions. This will involve fitting on an overdetermined system
			auto new_curve = curve.fit(t_params, boundary_points, curve.get_control_points().rows(), curve.get_knots());
			control_points = new_curve.get_control_points();
		}
		else
		{
			control_points = curve.get_control_points();
		}
	}

	void BSplineParametrization2D::derivative_wrt_params(const Eigen::VectorXd &grad_boundary, Eigen::VectorXd &grad_control_points)
	{
		grad_control_points.setZero(curve.get_control_points().size());
		nanospline::BSpline<double, 1, 3> curve_;
		curve_.set_knots(curve.get_knots());

		for (int i = 0; i < curve.get_control_points().rows(); ++i)
		{
			Eigen::MatrixXd indicator = Eigen::MatrixXd::Zero(curve.get_control_points().rows(), 1);
			indicator(i) = 1;
			curve_.set_control_points(indicator);
			for (const auto &b : node_ids_)
			{
				auto basis_val = curve_.evaluate(node_id_to_t_.at(b))(0);
				for (int k = 0; k < dim; ++k)
					grad_control_points(i * dim + k) += grad_boundary(b * dim + k) * basis_val;
			}
		}
	}

	void BSplineParametrization2D::gradient(const Eigen::MatrixXd &point, const Eigen::MatrixXd &control_points, const double t_parameter, const double distance, Eigen::MatrixXd &grad)
	{
		nanospline::BSpline<double, 2, 3> curve;
		curve.set_control_points(control_points);
		auto val = curve.evaluate(t_parameter);

		grad = (point - val) / distance;
	}

	void BSplineParametrization2D::eval(const Eigen::MatrixXd &control_points, const double t, Eigen::MatrixXd &val)
	{
		nanospline::BSpline<double, 2, 3> curve;
		curve.set_control_points(control_points);
		val = curve.evaluate(t);
	}

	void BSplineParametrization2D::deriv(const Eigen::MatrixXd &control_points, const double t, Eigen::MatrixXd &val)
	{
		nanospline::BSpline<double, 2, 3> curve;
		curve.set_control_points(control_points);
		val = curve.evaluate_derivative(t);
	}

	BSplineParametrization3D::BSplineParametrization3D(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &knots_u, const Eigen::MatrixXd &knots_v, const Eigen::MatrixXd &V) : BSplineParametrization(V), dim(control_points.cols())
	{
		assert(dim == 3);
		// Deduce the t parameter of all of the points in the spline sections
		double tol = 1e-4;
		patch.set_control_grid(control_points);
		patch.set_knots_u(knots_u);
		patch.set_knots_v(knots_v);
		patch.initialize();
		std::vector<int> unused;
		for (int i = 0; i < V.rows(); ++i)
			node_ids_.push_back(i);
		for (const auto &b : node_ids_)
		{
			Eigen::MatrixXd point = V.block(b, 0, 1, dim);
			auto uv = patch.approximate_inverse_evaluate(point, 5, 5, 0, 1, 0, 1, 30);
			assert(uv.size() == 2);
			double distance = (point - patch.evaluate(uv(0), uv(1))).norm();

			if (distance > tol)
			{
				logger().error("Could not find a valid uv for deducing spline parametrization. Distance: {}", distance);
				unused.push_back(b);
				continue;
			}

			node_id_to_param_[b] = uv;
		}

		if (unused.size() > 0)
			log_and_throw_error("Some nodes do not take part in the spline parametrization!");

		// Remove nodes that do not have a parametrization.
		// for (const auto &i : unused)
		// {
		// 	auto loc = std::find(node_ids_.begin(), node_ids_.end(), i);
		// 	if (loc == node_ids_.end())
		// 		logger().error("Error removing unused node.");
		// 	node_ids_.erase(loc);
		// }
		logger().info("Number of useful boundary nodes in spline parametrization: {}", node_ids_.size());
	}

	void BSplineParametrization3D::get_parameters(const Eigen::MatrixXd &V, Eigen::MatrixXd &control_points, const bool mesh_changed)
	{
		assert(!mesh_changed);
		Eigen::MatrixXd boundary_points(node_ids_.size(), dim), uv_params(node_ids_.size(), 2);
		bool boundary_changed = false;
		for (int i = 0; i < node_ids_.size(); ++i)
		// for (const auto &b : node_ids_)
		{
			auto uv = node_id_to_param_.at(node_ids_[i]);
			assert(uv.size() == 2);
			auto old_point = patch.evaluate(uv(0), uv(1));
			auto new_point = V.block(node_ids_[i], 0, 1, dim);
			boundary_points.block(i, 0, 1, dim) = new_point;
			uv_params.row(i) = uv;
			double difference = (old_point - new_point).norm();
			if (difference > 1e-4)
			{
				boundary_changed = true;
			}
		}
		if (boundary_changed)
		{
			// Deduce parameter values from vertex positions. This will involve fitting on an overdetermined system
			auto new_patch = patch.fit(uv_params, boundary_points, patch.get_knots_u().size() - 1 - 3, patch.get_knots_v().size() - 1 - 3, patch.get_knots_u(), patch.get_knots_v());
			control_points = new_patch.get_control_grid();
		}
		else
		{
			control_points = patch.get_control_grid();
		}
	}

	void BSplineParametrization3D::reparametrize(const Eigen::MatrixXd &control_points, Eigen::MatrixXd &newV)
	{
		// Given new control parameters and the t parameter precomputed, compute new V
		newV.setZero(node_ids_.size(), 3);
		patch.set_control_grid(control_points);
		patch.initialize();
		for (const auto &b : node_ids_)
		{
			auto uv = node_id_to_param_.at(b);
			assert(uv.size() == 2);
			auto new_val = patch.evaluate(uv(0), uv(1));
			newV.block(b, 0, 1, dim) = new_val;
		}
	}

	void BSplineParametrization3D::derivative_wrt_params(const Eigen::VectorXd &grad_boundary, Eigen::VectorXd &grad_control_points)
	{

		grad_control_points.setZero(patch.get_control_grid().size());

		for (int i = 0; i < patch.get_control_grid().rows(); ++i)
		{
			nanospline::BSplinePatch<double, 3, 3, 3> patch_;
			patch_.set_knots_u(patch.get_knots_u());
			patch_.set_knots_v(patch.get_knots_v());
			Eigen::MatrixXd indicator = Eigen::MatrixXd::Zero(patch.get_control_grid().rows(), 3);
			indicator.row(i) = Eigen::VectorXd::Ones(3);
			patch_.set_control_grid(indicator);
			patch_.initialize();
			for (const auto &b : node_ids_)
			{
				auto uv = node_id_to_param_.at(b);
				assert(uv.size() == 2);
				auto basis_val = patch_.evaluate(uv(0), uv(1))(0);
				for (int k = 0; k < dim; ++k)
					grad_control_points(i * dim + k) += grad_boundary(b * dim + k) * basis_val;
			}
		}
	}

	void BSplineParametrization3D::gradient(const Eigen::MatrixXd &point, const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &uv_parameter, const double distance, Eigen::MatrixXd &grad)
	{
		assert(uv_parameter.size() == 2);
		nanospline::BSplinePatch<double, 3, 3, 3> patch;
		patch.set_control_grid(control_points);
		auto val = patch.evaluate(uv_parameter(0), uv_parameter(1));

		grad = (point - val) / distance;
	}

	void BSplineParametrization3D::eval(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &uv_parameter, Eigen::MatrixXd &val)
	{
		assert(uv_parameter.size() == 2);
		nanospline::BSplinePatch<double, 3, 3, 3> patch;
		patch.set_control_grid(control_points);
		val = patch.evaluate(uv_parameter(0), uv_parameter(1));
	}

	void BSplineParametrization3D::deriv(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &uv_parameter, Eigen::MatrixXd &deriv_u, Eigen::MatrixXd &deriv_v)
	{
		assert(uv_parameter.size() == 2);
		nanospline::BSplinePatch<double, 3, 3, 3> patch;
		patch.set_control_grid(control_points);
		deriv_u = patch.evaluate_derivative_u(uv_parameter(0), uv_parameter(1));
		deriv_v = patch.evaluate_derivative_v(uv_parameter(0), uv_parameter(1));
	}
} // namespace polyfem