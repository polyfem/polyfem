#pragma once

#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <Eigen/Dense>

#include <nanospline/BSpline.h>

namespace polyfem
{
	class BSplineParametrization
	{
	public:
		BSplineParametrization(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &knots, const int boundary_id, const std::vector<int> &node_ids, const Eigen::MatrixXd &V) : boundary_id_(boundary_id), node_ids_(node_ids), dim(control_points.cols())
		{
			assert(dim == 2); // for now
			// Deduce the t parameter of all of the points in the spline sections
			double tol = 1e-4;
			curve.set_control_points(control_points);
			curve.set_knots(knots);
			std::vector<int> unused;
			for (const auto &b : node_ids)
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

			// Remove nodes that do not have a parametrization.
			for (const auto &i : unused)
			{
				auto loc = std::find(node_ids_.begin(), node_ids_.end(), i);
				if (loc == node_ids_.end())
					logger().error("Error removing unused node.");
				node_ids_.erase(loc);
			}
			logger().info("Number of useful boundary nodes in spline parametrization: {}", node_ids_.size());
		}

		void reparametrize(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &V, Eigen::MatrixXd &newV)
		{
			// Given new control parameters and the t parameter precomputed, compute new V
			curve.set_control_points(control_points);
			newV = V;
			for (const auto &b : node_ids_)
			{
				auto new_val = curve.evaluate(node_id_to_t_.at(b));
				newV.block(b, 0, 1, dim) = new_val;
			}
		}

		// Assume the connectivity has not changed. Does not work with remeshing
		void get_parameters(const Eigen::MatrixXd &V, Eigen::MatrixXd &control_points) const
		{
			Eigen::MatrixXd boundary_points(node_ids_.size(), dim), t_params(node_ids_.size(), 1);
			bool boundary_changed = false;
			for (int i = 0; i < node_ids_.size(); ++i)
			// for (const auto &b : node_ids_)
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

		void derivative_wrt_params(const Eigen::VectorXd &grad_boundary, Eigen::VectorXd &grad_control_points) const
		{
			grad_control_points.setZero(curve.get_control_points().size());
			nanospline::BSpline<double, 2, 3> curve_;
			curve_.set_knots(curve.get_knots());
			for (const auto &b : node_ids_)
			{
				for (int i = 0; i < curve.get_control_points().rows(); ++i)
				{
					Eigen::MatrixXd indicator = Eigen::MatrixXd::Zero(curve.get_control_points().rows(), dim);
					indicator.row(i) = Eigen::VectorXd::Ones(dim);
					curve_.set_control_points(indicator);
					auto basis_val = curve_.evaluate(node_id_to_t_.at(b));
					for (int k = 0; k < dim; ++k)
						grad_control_points(i * dim + k) += basis_val(k) * grad_boundary(b * dim + k);
				}
			}
		}

		static void gradient(const Eigen::MatrixXd &point, const Eigen::MatrixXd &control_points, const double t_parameter, const double distance, Eigen::MatrixXd &grad)
		{
			nanospline::BSpline<double, 2, 3> curve;
			curve.set_control_points(control_points);
			auto val = curve.evaluate(t_parameter);

			grad = (point - val) / distance;
		}

		static void eval(const Eigen::MatrixXd &control_points, const double t, Eigen::MatrixXd &val)
		{
			nanospline::BSpline<double, 2, 3> curve;
			curve.set_control_points(control_points);
			val = curve.evaluate(t);
		}

		static void deriv(const Eigen::MatrixXd &control_points, const double t, Eigen::MatrixXd &val)
		{
			nanospline::BSpline<double, 2, 3> curve;
			curve.set_control_points(control_points);
			val = curve.evaluate_derivative(t);
		}

		static void second_deriv(const Eigen::MatrixXd &control_points, const double t, Eigen::MatrixXd &val)
		{
			nanospline::BSpline<double, 2, 3> curve;
			curve.set_control_points(control_points);
			val = curve.evaluate_2nd_derivative(t);
		}

	private:
		int boundary_id_;
		std::vector<int> node_ids_;
		std::map<int, double> node_id_to_t_;
		const int dim;
		nanospline::BSpline<double, 2, 3> curve;
	};
} // namespace polyfem