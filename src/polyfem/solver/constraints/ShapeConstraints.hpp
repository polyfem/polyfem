#pragma once

#include "Constraints.hpp"

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
			state->set_v(V_full);
		}

		ShapeConstraints(const json &constraint_params, const int num_vertices, const int dim)
			: Constraints(constraint_params, num_vertices * dim, 0), num_vertices_(num_vertices), dim_(dim)
		{
			int reduced_size;
			if (!constraint_params.contains("restriction") || constraint_params["restriction"] == "none")
			{
				reduced_size_ = num_vertices * dim;
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
					for (int b : fixed_nodes_)
						for (int d = 0; d < dim_; d++)
							dreduced(b * dim_ + d) = 0;
					return dreduced;
				};
			}
		}

		void set_fixed_nodes(const std::set<int> &fixed_nodes)
		{
			fixed_nodes_ = fixed_nodes;
		}

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
		std::set<int> fixed_nodes_;
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