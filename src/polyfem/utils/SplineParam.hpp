#include <map>
#include <vector>
#include <Eigen/Dense>

#include <polysolve/LinearSolver.hpp>

namespace polyfem
{
	class SplineParam
	{
	public:
		SplineParam(const std::map<int, Eigen::MatrixXd> &control_point, const std::map<int, Eigen::MatrixXd> &tangent, const std::map<int, std::vector<int>> &boundary_id_to_node, const Eigen::MatrixXd &V, const int sampling) : boundary_id_to_node_(boundary_id_to_node)
		{
			// Deduce the t parameter of all of the points in the spline sections
			for (const auto &kv : boundary_id_to_node)
			{
				boundary_id_to_t_[kv.first] = {};
				for (const auto &pt : kv.second)
				{
					// TODO replace with something more sensible, such as minimization, but for now just do this
					double t = 0;
					double dt = 1. / sampling;
					Eigen::MatrixXd val(2, 1);
					for (int i = 0; i < sampling; ++i)
					{
						eval(control_point.at(kv.first), tangent.at(kv.first), i * dt, val);
						if ((val - V.row(pt)).norm() < 1e-4)
						{
							t = i * dt;
							break;
						}
					}
					boundary_id_to_t_[kv.first].push_back(t);
				}
			}
		}

		void reparametrize(const std::map<int, Eigen::MatrixXd> &control_point, const std::map<int, Eigen::MatrixXd> &tangent, const Eigen::MatrixXd &V, Eigen::MatrixXd &newV)
		{
			// Given new control parameters and the t parameter precomputed, compute new V
			newV = V;
			for (const auto &kv : boundary_id_to_node_)
			{
				for (int i = 0; i < boundary_id_to_node_[kv.first].size(); ++i)
				{
					Eigen::MatrixXd new_val;
					eval(control_point.at(kv.first), tangent.at(kv.first), boundary_id_to_t_[kv.first][i], new_val);
					newV.row(boundary_id_to_node_[kv.first][i]) = new_val;
				}
			}
		}

		// Assume the connectivity has not changed. Does not work with remeshing
		void get_parameters(const Eigen::MatrixXd &V, std::map<int, Eigen::MatrixXd> &control_point, std::map<int, Eigen::MatrixXd> &tangent)
		{
			// Deduce parameter values from vertex positions. This will involve fitting on an overdetermined system
			for (const auto &kv : boundary_id_to_node_)
			{
				Eigen::MatrixXd A;
				Eigen::VectorXd b;
				A.setZero(2 * boundary_id_to_node_.size(), 8);
				b.setZero(2 * boundary_id_to_node_.size());
				for (int i = 0; i < boundary_id_to_node_[kv.first].size(); ++i)
				{
					int index = boundary_id_to_node_[kv.first][i];
					double t = boundary_id_to_t_[kv.first][i];
					double t_2 = pow(t, 2);
					double t_3 = pow(t, 3);
					for (int dim = 0; dim < 2; ++dim)
					{
						A(2 * i + dim, dim) = 2 * t_3 - 3 * t_2 + 1;
						A(2 * i + dim, 2 + dim) = t_3 - 2 * t_2 + t;
						A(2 * i + dim, 4 + dim) = -2 * t_3 + 3 * t_2;
						A(2 * i + dim, 6 + dim) = t_3 - t_2;
						b(2 * i + dim) = V(index, dim);
					}
				}

				Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
				Eigen::MatrixXd control_points(2, 2), tangents(2, 2);
				control_points.row(0) = x.segment(0, 2);
				tangents.row(0) = x.segment(2, 2);
				control_points.row(1) = x.segment(4, 2);
				tangents.row(1) = x.segment(6, 2);
				control_point[kv.first] = control_points;
				tangent[kv.first] = tangents;
			}
		}

		void derivative_wrt_params(const Eigen::VectorXd &grad_boundary, const int boundary_id, Eigen::VectorXd &grad_control_point, Eigen::VectorXd &grad_tangent)
		{
			grad_control_point.setZero(4);
			grad_tangent.setZero(4);
			for (int i = 0; i < boundary_id_to_node_[boundary_id].size(); ++i)
			{
				double t = boundary_id_to_t_[boundary_id][i];
				double t_2 = pow(t, 2);
				double t_3 = pow(t, 3);
				grad_control_point.segment(0, 2) += (2 * t_3 - 3 * t_2 + 1) * grad_boundary.segment(boundary_id_to_node_[boundary_id][i] * 2, 2);
				grad_control_point.segment(2, 2) += (-2 * t_3 + 3 * t_2) * grad_boundary.segment(boundary_id_to_node_[boundary_id][i] * 2, 2);
				grad_tangent.segment(0, 2) += (t_3 - 2 * t_2 + t) * grad_boundary.segment(boundary_id_to_node_[boundary_id][i] * 2, 2);
				grad_tangent.segment(2, 2) += (t_3 - t_2) * grad_boundary.segment(boundary_id_to_node_[boundary_id][i] * 2, 2);
			}
		}

	private:
		void eval(const Eigen::MatrixXd &control_point, const Eigen::MatrixXd &tangent, const double t, Eigen::MatrixXd &val)
		{
			double t_2 = pow(t, 2);
			double t_3 = pow(t, 3);
			val = (2 * t_3 - 3 * t_2 + 1) * control_point.row(0);
			val += (t_3 - 2 * t_2 + t) * tangent.row(0);
			val += (-2 * t_3 + 3 * t_2) * control_point.row(1);
			val += (t_3 - t_2) * tangent.row(1);
		}

		std::map<int, std::vector<int>> boundary_id_to_node_;
		std::map<int, std::vector<int>> boundary_id_to_t_;
	};
} // namespace polyfem