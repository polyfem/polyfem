#pragma once

#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <Eigen/Dense>

#include <polysolve/LinearSolver.hpp>

namespace polyfem
{
	class CubicHermiteSplineParametrization
	{
	public:
		CubicHermiteSplineParametrization(const std::map<int, Eigen::MatrixXd> &control_point, const std::map<int, Eigen::MatrixXd> &tangent, const std::map<int, std::vector<int>> &boundary_id_to_node_id, const Eigen::MatrixXd &V, const int sampling) : boundary_id_to_node_id_(boundary_id_to_node_id)
		{
			// Deduce the t parameter of all of the points in the spline sections
			int index = 0;
			double tol = 1e-4;
			for (const auto &kv : boundary_id_to_node_id_)
			{
				std::vector<int> unused;
				boundary_id_to_spline_count_[kv.first] = control_point.at(kv.first).rows() - 1;
				for (const auto &b : kv.second)
				{
					Eigen::MatrixXd point = V.block(b, 0, 1, dim);
					point.transposeInPlace();

					int nearest;
					double t_optimal, distance, distance_to_start, distance_to_end;
					find_nearest_spline(point, control_point.at(kv.first), tangent.at(kv.first), nearest, t_optimal, distance, distance_to_start, distance_to_end);

					if (nearest == -1 || distance > tol)
					{
						logger().error("Could not find a valid t for deducing spline parametrization. Distance: {}, spline: {}, point: {}, {}", distance, nearest, point(0), point(1));
						unused.push_back(b);
						continue;
					}

					node_id_to_t_[b] = t_optimal;
					node_id_to_spline_[b] = nearest;
				}

				// Remove nodes that do not have a parametrization.
				for (const auto &i : unused)
				{
					auto loc = std::find(boundary_id_to_node_id_[kv.first].begin(), boundary_id_to_node_id_[kv.first].end(), i);
					if (loc == boundary_id_to_node_id_[kv.first].end())
						logger().error("Error removing unused node.");
					boundary_id_to_node_id_[kv.first].erase(loc);
				}
				logger().info("Number of useful boundary nodes in spline parametrization: {}", boundary_id_to_node_id_[kv.first].size());
			}
		}

		void reparametrize(const std::map<int, Eigen::MatrixXd> &control_point, const std::map<int, Eigen::MatrixXd> &tangent, const Eigen::MatrixXd &V, Eigen::MatrixXd &newV) const
		{
			// Given new control parameters and the t parameter precomputed, compute new V
			newV = V.block(0, 0, V.rows(), 2);
			for (const auto &kv : boundary_id_to_node_id_)
			{
				for (const auto &b : kv.second)
				{
					Eigen::MatrixXd new_val;
					eval(control_point.at(kv.first).block(node_id_to_spline_.at(b), 0, 2, dim), tangent.at(kv.first).block(2 * node_id_to_spline_.at(b), 0, 2, dim), node_id_to_t_.at(b), new_val);
					newV.block(b, 0, 1, dim) = new_val;
				}
			}
		}

		// Assume the connectivity has not changed. Does not work with remeshing
		void get_parameters(const Eigen::MatrixXd &V, std::map<int, Eigen::MatrixXd> &control_point, std::map<int, Eigen::MatrixXd> &tangent) const
		{
			// Deduce parameter values from vertex positions. This will involve fitting on an overdetermined system
			for (const auto &kv : boundary_id_to_node_id_)
			{
				int spline_count = boundary_id_to_spline_count_.at(kv.first);
				Eigen::MatrixXd A;
				Eigen::MatrixXd b;
				A.setZero(kv.second.size(), 3 * spline_count + 1);
				b.setZero(kv.second.size(), 2);
				int i = 0;
				for (const int id : kv.second)
				{
					double t = node_id_to_t_.at(id);
					double t_2 = pow(t, 2);
					double t_3 = pow(t, 3);
					int spline = node_id_to_spline_.at(id);
					A(i, spline) = 2 * t_3 - 3 * t_2 + 1;
					A(i, spline + 1) = -2 * t_3 + 3 * t_2;
					A(i, spline_count + 1 + 2 * spline) = t_3 - 2 * t_2 + t;
					A(i, spline_count + 1 + 2 * spline + 1) = t_3 - t_2;
					b(i, 0) = V(id, 0);
					b(i, 1) = V(id, 1);
					i++;
				}

				Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b.col(0));
				Eigen::VectorXd y = (A.transpose() * A).ldlt().solve(A.transpose() * b.col(1));
				double error_x = (A * x - b.col(0)).norm();
				double error_y = (A * y - b.col(1)).norm();
				Eigen::MatrixXd control_points(spline_count + 1, dim), tangents(2 * spline_count, dim);
				control_points.col(0) = x.segment(0, spline_count + 1);
				control_points.col(1) = y.segment(0, spline_count + 1);
				tangents.col(0) = x.segment(spline_count + 1, 2 * spline_count);
				tangents.col(1) = y.segment(spline_count + 1, 2 * spline_count);
				control_point[kv.first] = control_points;
				tangent[kv.first] = tangents;
			}
		}

		void derivative_wrt_params(const Eigen::VectorXd &grad_boundary, const int boundary_id, const int couple_tangents, Eigen::VectorXd &grad_control_point, Eigen::VectorXd &grad_tangent) const
		{
			int spline_count = boundary_id_to_spline_count_.at(boundary_id);
			grad_control_point.setZero(spline_count * dim + dim);
			grad_tangent.setZero(spline_count * 2 * dim);
			for (const auto &b : boundary_id_to_node_id_.at(boundary_id))
			{
				double t = node_id_to_t_.at(b);
				double t_2 = pow(t, 2);
				double t_3 = pow(t, 3);
				int spline = node_id_to_spline_.at(b);
				grad_control_point.segment(spline * dim, dim) += (2 * t_3 - 3 * t_2 + 1) * grad_boundary.segment(b * dim, dim);
				grad_control_point.segment(spline * dim + dim, dim) += (-2 * t_3 + 3 * t_2) * grad_boundary.segment(b * dim, dim);
				grad_tangent.segment(spline * 2 * dim, dim) += (t_3 - 2 * t_2 + t) * grad_boundary.segment(b * dim, dim);
				grad_tangent.segment(spline * 2 * dim + dim, dim) += (t_3 - t_2) * grad_boundary.segment(b * dim, dim);
				if (spline != 0)
				{
					int prev_spline = spline - 1;
					if (couple_tangents)
						grad_tangent.segment(prev_spline * 2 * dim + dim, dim) += (t_3 - 2 * t_2 + t) * grad_boundary.segment(b * dim, dim);
					else
					{
						// The following is not correct, need to project onto normal of tangent.
						grad_tangent.segment(prev_spline * 2 * dim + dim, dim) += (t_3 - 2 * t_2 + t) * grad_boundary.segment(b * dim, dim);
					}
				}
				if (spline != spline_count - 1)
				{
					int next_spline = spline + 1;
					if (couple_tangents)
						grad_tangent.segment(next_spline * 2 * dim, dim) += (t_3 - t_2) * grad_boundary.segment(b * dim, dim);
					else
					{
						// The following is not correct, need to project onto normal of tangent.
						grad_tangent.segment(next_spline * 2 * dim, dim) += (t_3 - t_2) * grad_boundary.segment(b * dim, dim);
					}
				}
			}
		}

		static void find_nearest_spline(const Eigen::MatrixXd &point, const Eigen::MatrixXd &control_point, const Eigen::MatrixXd &tangent, int &nearest, double &t_optimal, double &distance, double &distance_to_start, double &distance_to_end, const double tol = 1e-4)
		{
			int dim = point.size();
			int num_splines = control_point.rows() - 1;

			auto dot = [](const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
				assert(a.cols() == 1);
				assert(b.cols() == 1);
				return (a.transpose() * b)(0);
			};

			auto f = [&](const double t, const int segment) {
				Eigen::MatrixXd val;
				eval(control_point.block(segment, 0, 2, dim), tangent.block(2 * segment, 0, 2, dim), t, val);
				val.transposeInPlace();
				return val;
			};

			auto f_ = [&](const double t, const int segment) {
				Eigen::MatrixXd val;
				deriv(control_point.block(segment, 0, 2, dim), tangent.block(2 * segment, 0, 2, dim), t, val);
				val.transposeInPlace();
				return val;
			};

			auto f__ = [&](const double t, const int segment) {
				Eigen::MatrixXd val;
				second_deriv(control_point.block(segment, 0, 2, dim), tangent.block(2 * segment, 0, 2, dim), t, val);
				val.transposeInPlace();
				return val;
			};

			auto g = [&](const double t, const int segment) {
				auto val = f(t, segment);
				return dot(val, val) - 2 * dot(point, val) + dot(point, point);
			};

			auto g_ = [&](const double t, const int segment) {
				auto val = f(t, segment);
				auto deriv = f_(t, segment);
				return 2 * dot(deriv, val) - 2 * dot(point, deriv);
			};

			auto g__ = [&](const double t, const int segment) {
				auto val = f(t, segment);
				auto deriv = f_(t, segment);
				auto sec_deriv = f__(t, segment);
				return 2 * dot(sec_deriv, val - point) + 2 * dot(deriv, deriv);
			};

			Eigen::MatrixXd t = Eigen::MatrixXd::Ones(num_splines, 1) / 2.;

			for (int i = 0; i < t.rows(); ++i)
			{
				for (int iter = 0; iter < 50; ++iter)
				{
					t(i) -= g_(t(i), i) / g__(t(i), i);
				}
			}

			nearest = -1;
			t_optimal = -1;
			distance = -1.;
			for (int i = 0; i < t.rows(); ++i)
			{
				if ((t(i) < -tol) || (t(i) > 1 + tol))
					continue;
				double func = g(t(i), i);
				if ((nearest == -1) || (func < distance))
				{
					nearest = i;
					t_optimal = t(i);
					distance = func;
				}
			}

			distance_to_start = g(0, 0);
			distance_to_end = g(1, t.rows() - 1);
		}

		static void gradient(const Eigen::MatrixXd &point, const Eigen::MatrixXd &control_point, const Eigen::MatrixXd &tangent, const int spline, const double t_parameter, const double distance, Eigen::MatrixXd &grad)
		{
			int dim = point.size();

			auto f = [&](const double t, const int segment) {
				Eigen::MatrixXd val;
				eval(control_point.block(segment, 0, 2, dim), tangent.block(2 * segment, 0, 2, dim), t, val);
				val.transposeInPlace();
				return val;
			};

			grad.col(0).segment(0, dim) = (point - f(t_parameter, spline)) / distance;
		}

		static void eval(const Eigen::MatrixXd &control_point, const Eigen::MatrixXd &tangent, const double t, Eigen::MatrixXd &val)
		{
			double t_2 = pow(t, 2);
			double t_3 = pow(t, 3);
			val = (2 * t_3 - 3 * t_2 + 1) * control_point.row(0);
			val += (t_3 - 2 * t_2 + t) * tangent.row(0);
			val += (-2 * t_3 + 3 * t_2) * control_point.row(1);
			val += (t_3 - t_2) * tangent.row(1);
		}

		static void deriv(const Eigen::MatrixXd &control_point, const Eigen::MatrixXd &tangent, const double t, Eigen::MatrixXd &val)
		{
			double t_2 = pow(t, 2);
			val = (6 * t_2 - 6 * t) * control_point.row(0);
			val += (3 * t_2 - 4 * t + 1) * tangent.row(0);
			val += (-6 * t_2 + 6 * t) * control_point.row(1);
			val += (3 * t_2 - 2 * t) * tangent.row(1);
		}

		static void second_deriv(const Eigen::MatrixXd &control_point, const Eigen::MatrixXd &tangent, const double t, Eigen::MatrixXd &val)
		{
			val = (12 * t - 6) * control_point.row(0);
			val += (6 * t - 4) * tangent.row(0);
			val += (-12 * t + 6) * control_point.row(1);
			val += (6 * t - 2) * tangent.row(1);
		}

	private:
		std::map<int, std::vector<int>> boundary_id_to_node_id_;
		std::map<int, double> node_id_to_t_;
		std::map<int, int> node_id_to_spline_;
		int dim = 2;
		std::map<int, int> boundary_id_to_spline_count_;
	};
} // namespace polyfem