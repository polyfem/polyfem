#include <map>
#include <vector>
#include <set>
#include <Eigen/Dense>

#include <polysolve/LinearSolver.hpp>

namespace polyfem
{
	class SplineParam
	{
	public:
		SplineParam(const std::map<int, Eigen::MatrixXd> &control_point, const std::map<int, Eigen::MatrixXd> &tangent, const std::map<int, std::vector<int>> &boundary_id_to_node_id, const Eigen::MatrixXd &V, const int sampling) : boundary_id_to_node_id_(boundary_id_to_node_id)
		{
			// Deduce the t parameter of all of the points in the spline sections
			int index = 0;
			for (const auto &kv : boundary_id_to_node_id_)
			{
				std::vector<int> nodes(kv.second.begin(), kv.second.end());
				auto g = [&](const Eigen::MatrixXd &t) {
					Eigen::MatrixXd fun(2 * t.rows(), 1);
					for (int i = 0; i < t.rows(); ++i)
					{
						Eigen::MatrixXd val;
						eval(control_point.at(kv.first), tangent.at(kv.first), t(i, 0), val);
						fun(2 * i, 0) = val(0, 0) - V(nodes[i], 0);
						fun(2 * i + 1, 0) = val(0, 1) - V(nodes[i], 1);
					}
					return fun;
				};
				auto J = [&](const Eigen::VectorXd &t) {
					Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(2 * t.rows(), t.rows());
					for (int i = 0; i < t.rows(); ++i)
					{
						Eigen::MatrixXd val;
						deriv(control_point.at(kv.first), tangent.at(kv.first), t(i, 0), val);
						jac(2 * i, i) = val(0, 0);
						jac(2 * i + 1, i) = val(0, 1);
					}
					return jac;
				};

				Eigen::MatrixXd t = Eigen::MatrixXd::Ones(nodes.size(), 1) / 2.;
				for (int i = 0; i < 20; ++i)
				{
					Eigen::MatrixXd jac_inv = J(t).completeOrthogonalDecomposition().pseudoInverse();
					Eigen::MatrixXd func = g(t);
					t -= jac_inv * func;
				}

				for (int i = 0; i < nodes.size(); ++i)
					node_id_to_t_[nodes[i]] = t(i, 0);
			}
		}

		void reparametrize(const std::map<int, Eigen::MatrixXd> &control_point, const std::map<int, Eigen::MatrixXd> &tangent, const Eigen::MatrixXd &V, Eigen::MatrixXd &newV) const
		{
			// Given new control parameters and the t parameter precomputed, compute new V
			newV = V.block(0, 0, V.rows(), 2);
			for (const auto &kv : boundary_id_to_node_id_)
			{
				for (const int id : kv.second)
				{
					Eigen::MatrixXd new_val;
					eval(control_point.at(kv.first), tangent.at(kv.first), node_id_to_t_.at(id), new_val);
					newV.block(id, 0, 1, 2) = new_val;
				}
			}
		}

		// Assume the connectivity has not changed. Does not work with remeshing
		void get_parameters(const Eigen::MatrixXd &V, std::map<int, Eigen::MatrixXd> &control_point, std::map<int, Eigen::MatrixXd> &tangent) const
		{
			// Deduce parameter values from vertex positions. This will involve fitting on an overdetermined system
			for (const auto &kv : boundary_id_to_node_id_)
			{
				Eigen::MatrixXd A;
				Eigen::MatrixXd b;
				A.setZero(kv.second.size(), 4);
				b.setZero(kv.second.size(), 2);
				int i = 0;
				for (const int id : kv.second)
				{
					double t = node_id_to_t_.at(id);
					double t_2 = pow(t, 2);
					double t_3 = pow(t, 3);
					A(i, 0) = 2 * t_3 - 3 * t_2 + 1;
					A(i, 1) = -2 * t_3 + 3 * t_2;
					A(i, 2) = t_3 - 2 * t_2 + t;
					A(i, 3) = t_3 - t_2;
					b(i, 0) = V(id, 0);
					b(i, 1) = V(id, 1);
					i++;
				}

				Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b.col(0));
				Eigen::VectorXd y = (A.transpose() * A).ldlt().solve(A.transpose() * b.col(1));
				double error_x = (A * x - b.col(0)).norm();
				double error_y = (A * y - b.col(1)).norm();
				Eigen::MatrixXd control_points(2, 2), tangents(2, 2);
				control_points.col(0) = x.segment(0, 2);
				control_points.col(1) = y.segment(0, 2);
				tangents.col(0) = x.segment(2, 2);
				tangents.col(1) = y.segment(2, 2);
				control_point[kv.first] = control_points;
				tangent[kv.first] = tangents;
			}
		}

		void derivative_wrt_params(const Eigen::VectorXd &grad_boundary, const int boundary_id, Eigen::VectorXd &grad_control_point, Eigen::VectorXd &grad_tangent) const
		{
			grad_control_point.setZero(4);
			grad_tangent.setZero(4);
			for (const int id : boundary_id_to_node_id_.at(boundary_id))
			{
				double t = node_id_to_t_.at(id);
				double t_2 = pow(t, 2);
				double t_3 = pow(t, 3);
				grad_control_point.segment(0, 2) += (2 * t_3 - 3 * t_2 + 1) * grad_boundary.segment(id * 2, 2);
				grad_control_point.segment(2, 2) += (-2 * t_3 + 3 * t_2) * grad_boundary.segment(id * 2, 2);
				grad_tangent.segment(0, 2) += (t_3 - 2 * t_2 + t) * grad_boundary.segment(id * 2, 2);
				grad_tangent.segment(2, 2) += (t_3 - t_2) * grad_boundary.segment(id * 2, 2);
			}
		}

	private:
		void eval(const Eigen::MatrixXd &control_point, const Eigen::MatrixXd &tangent, const double t, Eigen::MatrixXd &val) const
		{
			double t_2 = pow(t, 2);
			double t_3 = pow(t, 3);
			val = (2 * t_3 - 3 * t_2 + 1) * control_point.row(0);
			val += (t_3 - 2 * t_2 + t) * tangent.row(0);
			val += (-2 * t_3 + 3 * t_2) * control_point.row(1);
			val += (t_3 - t_2) * tangent.row(1);
		}

		void deriv(const Eigen::MatrixXd &control_point, const Eigen::MatrixXd &tangent, const double t, Eigen::MatrixXd &val) const
		{
			double t_2 = pow(t, 2);
			val = (6 * t_2 - 6 * t) * control_point.row(0);
			val += (3 * t_2 - 4 * t + 1) * tangent.row(0);
			val += (-6 * t_2 + 6 * t) * control_point.row(1);
			val += (3 * t_2 - 2 * t) * tangent.row(1);
		}

		const std::map<int, std::vector<int>> boundary_id_to_node_id_;
		std::map<int, double> node_id_to_t_;
	};
} // namespace polyfem