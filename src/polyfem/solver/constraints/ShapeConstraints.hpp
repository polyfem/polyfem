#pragma once

#include "Constraints.hpp"

#include <polyfem/utils/CompositeSplineParam.hpp>

namespace polyfem
{
	class ShapeConstraints : public Constraints
	{
	public:
		void update_state(std::shared_ptr<State> state, const Eigen::VectorXd &reduced) override
		{
			Eigen::MatrixXd V_rest, V_full;
			Eigen::MatrixXi F;
			state->get_vf(V_rest, F);
			reduced_to_full(reduced, V_rest, V_full);
			state->set_mesh_vertices(V_full);
			state->build_basis();
		}

		ShapeConstraints(const json &constraint_params, const Eigen::MatrixXd &V_start, const std::map<int, std::vector<int>> &optimization_boundary_to_node_ids)
			: Constraints(constraint_params, V_start.size(), 0), num_vertices_(V_start.rows()), dim_(V_start.cols()), optimization_boundary_to_node_ids_(optimization_boundary_to_node_ids)
		{
			std::string restriction = constraint_params["restriction"];
			if (restriction == "none")
			{
				reduced_size_ = num_vertices_ * dim_;
				reduced_to_full_ = [this](const Eigen::VectorXd &reduced, const Eigen::MatrixXd &V_rest) {
					Eigen::MatrixXd V(num_vertices_, dim_);
					for (int i = 0; i < num_vertices_; i++)
						for (int d = 0; d < dim_; d++)
							V(i, d) = reduced(i * dim_ + d);
					return V;
				};
				full_to_reduced_ = [this](const Eigen::MatrixXd &V_full) {
					Eigen::VectorXd reduced(reduced_size_);
					for (int i = 0; i < num_vertices_; i++)
						for (int d = 0; d < dim_; d++)
							reduced(i * dim_ + d) = V_full(i, d);
					return reduced;
				};
				dfull_to_dreduced_ = [this](const Eigen::VectorXd &dV_full) {
					Eigen::VectorXd dreduced = dV_full;
					for (int b = 0; b < active_nodes_mask_.size(); b++)
						if (!active_nodes_mask_[b])
							for (int d = 0; d < dim_; d++)
								dreduced(b * dim_ + d) = 0;
					return dreduced;
				};
			}
			else if (restriction == "cubic_hermite_spline")
			{
				const auto &spline_params = constraint_params["spline_specification"];
				std::map<int, Eigen::MatrixXd> control_point, tangent;
				bool couple_tangents; // couple the direction and magnitude of adjacent tangents
				reduced_size_ = 0;
				assert(dim_ == 2);
				for (const auto &spline : spline_params)
				{
					const int boundary_id = spline["id"].get<int>();
					auto control_points = spline["control_point"];
					auto tangents = spline["tangent"];
					Eigen::MatrixXd c(control_points.size(), dim_), t(2 * control_points.size() - 2, dim_);
					if (control_points.size() == tangents.size())
					{
						couple_tangents = true;
						for (int i = 0; i < control_points.size(); ++i)
						{
							assert(control_points[i].size() == dim_);
							assert(tangents[i].size() == dim_);
							for (int k = 0; k < dim_; ++k)
							{
								c(i, k) = control_points[i][k];
								for (int j = 0; j < 2; ++j)
								{
									if (i != 0)
										t(2 * i - 1, k) = tangents[i][k];
									if (i != (control_points.size() - 1))
										t(2 * i, k) = tangents[i][k];
								}
							}
						}
					}
					else if ((2 * control_points.size() - 2) == tangents.size())
					{
						couple_tangents = false;
						for (int i = 0; i < control_points.size(); ++i)
						{
							assert(control_points[i].size() == dim_);
							for (int k = 0; k < dim_; ++k)
								c(i, k) = control_points[i][k];
						}
						for (int i = 0; i < tangents.size(); ++i)
						{
							assert(tangents[i].size() == dim_);
							for (int k = 0; k < dim_; ++k)
								t(i, k) = tangents[i][k];
						}
					}
					else
					{
						logger().error("The number of tangents must be either equal to (or twice of) number of control points.");
					}

					control_point.insert({boundary_id, c});
					tangent.insert({boundary_id, t});
					reduced_size_ += 2 * (c.rows() - 2);
					reduced_size_ += 2 * t.rows();
					logger().trace("Given tangents are: {}", t);
				}
				CompositeSplineParam spline_param(control_point, tangent, optimization_boundary_to_node_ids_, V_start, 0);
				full_to_reduced_ = [this, spline_param](const Eigen::MatrixXd &V_full) {
					Eigen::VectorXd reduced(reduced_size_);
					std::map<int, Eigen::MatrixXd> control_point, tangent;
					spline_param.get_parameters(V_full, control_point, tangent);
					int index = 0;
					for (const auto &kv : control_point)
					{
						for (int i = 0; i < kv.second.rows(); ++i)
						{
							if (i == 0 || i == (kv.second.rows() - 1))
								continue;
							reduced.segment(index, dim_) = kv.second.row(i);
							index += dim_;
						}
					}
					for (const auto &kv : tangent)
					{
						for (int i = 0; i < kv.second.rows(); ++i)
						{
							reduced.segment(index, dim_) = kv.second.row(i);
							index += dim_;
						}
					}
					assert(index == reduced.size());
					return reduced;
				};
				reduced_to_full_ = [this, control_point, tangent, spline_param](const Eigen::VectorXd &reduced, const Eigen::MatrixXd &V_rest) {
					Eigen::MatrixXd V(num_vertices_, dim_);
					std::map<int, Eigen::MatrixXd> new_control_point, new_tangent;
					int index = 0;
					for (const auto &kv : control_point)
					{
						Eigen::MatrixXd control_point_matrix(kv.second.rows(), kv.second.cols());
						for (int i = 0; i < kv.second.rows(); ++i)
						{
							if (i == 0 || i == (kv.second.rows() - 1))
								control_point_matrix.row(i) = kv.second.row(i);
							else
							{
								control_point_matrix.row(i) = reduced.segment(index, kv.second.cols());
								index += kv.second.cols();
							}
						}
						new_control_point[kv.first] = control_point_matrix;
					}
					for (const auto &kv : tangent)
					{
						Eigen::MatrixXd tangent_matrix(kv.second.rows(), kv.second.cols());
						for (int i = 0; i < kv.second.rows(); ++i)
						{
							tangent_matrix.row(i) = reduced.segment(index, kv.second.cols());
							index += kv.second.cols();
						}
						new_tangent[kv.first] = tangent_matrix;
					}
					spline_param.reparametrize(new_control_point, new_tangent, V_rest, V);
					return V;
				};
				dfull_to_dreduced_ = [this, control_point, spline_param, couple_tangents](const Eigen::VectorXd &dV_full) {
					Eigen::VectorXd dreduced;
					dreduced.setZero(reduced_size_);
					int index = 0;
					for (const auto &kv : control_point)
					{
						Eigen::VectorXd grad_control_point, grad_tangent;
						spline_param.derivative_wrt_params(dV_full, kv.first, couple_tangents, grad_control_point, grad_tangent);
						dreduced.segment(index, grad_control_point.rows() - 2 * dim_) = grad_control_point.segment(dim_, grad_control_point.rows() - 2 * dim_);
						index += grad_control_point.rows() - 2 * dim_;
						dreduced.segment(index, grad_tangent.rows()) = grad_tangent;
						index += grad_tangent.rows();
					}
					return dreduced;
				};
			}
			else if (restriction == "graph_structure")
			{
				const auto &graph_param = constraint_params["graph_specification"];
				assert(false);
			}
		}

		void set_active_nodes_mask(const std::vector<bool> &active_nodes_mask) { active_nodes_mask_ = active_nodes_mask; }

		void reduced_to_full(const Eigen::VectorXd &reduced, const Eigen::MatrixXd &V_rest, Eigen::MatrixXd &V_full) const
		{
			V_full = reduced_to_full_(reduced, V_rest);
		}

		void full_to_reduced(const Eigen::MatrixXd &V_full, Eigen::VectorXd &reduced) const
		{
			if (full_to_reduced_ != nullptr)
			{
				reduced = full_to_reduced_(V_full);
				return;
			}

			// DiffScalarBase::setVariableCount(full_size_); // size of input vector
			// assert(full.size() == full_size_);
			// DiffVector full_diff(full_size_);
			// for (int i = 0; i < full.size(); ++i)
			// 	full_diff(i) = DiffScalar(i, full(i));
			// auto reduced_diff = full_to_reduced_diff_(full_diff);
			// assert(reduced_diff.size() == reduced_size_);
			// reduced.resize(reduced_size_);
			// for (int i = 0; i < reduced_size_; ++i)
			// 	reduced(i) = reduced_diff(i).getValue();
		}

		void dfull_to_dreduced(const Eigen::VectorXd &V_full, const Eigen::VectorXd &dV_full, Eigen::VectorXd &dreduced) const
		{
			if (dfull_to_dreduced_ != nullptr)
			{
				dreduced = dfull_to_dreduced_(dV_full);
				return;
			}

			// DiffScalarBase::setVariableCount(full_size_); // size of input vector
			// assert(full.size() == full_size_);
			// DiffVector full_diff(full_size_);
			// for (int i = 0; i < full.size(); ++i)
			// 	full_diff(i) = DiffScalar(i, full(i));
			// auto reduced_diff = full_to_reduced_diff_(full_diff);
			// assert(reduced_diff.size() == reduced_size_);
			// Eigen::MatrixXd grad(reduced_size_, full_size_);
			// for (int i = 0; i < reduced_size_; ++i)
			// {
			// 	for (int j = 0; j < full.size(); ++j)
			// 		grad(i, j) = reduced_diff(i).getGradient()(j);
			// }

			// dreduced = grad * dV_full;
		}

	private:
		std::vector<bool> active_nodes_mask_;
		std::map<int, std::vector<int>> optimization_boundary_to_node_ids_;
		const int num_vertices_;
		const int dim_;

		// Must define the inverse function with Eigen types, differentiability is not needed.
		std::function<Eigen::MatrixXd(const Eigen::VectorXd &, const Eigen::MatrixXd &)> reduced_to_full_;

		// For differentiability, either define the function with differentiable types.
		// std::function<DiffVector(const DiffMatrix &)> full_to_reduced_diff_;
		// Or explicitly define the function and gradient with Eigen types.
		std::function<Eigen::VectorXd(const Eigen::MatrixXd &)> full_to_reduced_;
		std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> dfull_to_dreduced_;
	};
} // namespace polyfem