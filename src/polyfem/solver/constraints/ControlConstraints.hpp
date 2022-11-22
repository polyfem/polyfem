#pragma once

#include "Constraints.hpp"

namespace polyfem
{
	class ControlConstraints : public Constraints
	{
	public:
		void update_state(std::shared_ptr<State> state, const Eigen::VectorXd &reduced) override
		{
		}

		ControlConstraints(const json &constraint_params, const int time_steps, const int dim, const std::vector<int> &boundary_ids_list, const std::map<int, int> &boundary_id_to_reduced_dim)
			: Constraints(constraint_params, boundary_ids_list_.size(), 0), time_steps_(time_steps), dim_(dim), boundary_ids_list_(boundary_ids_list), boundary_id_to_reduced_dim_(boundary_id_to_reduced_dim)
		{
			std::string restriction = constraint_params["restriction"];
			if (restriction == "none")
			{
				reduced_size_ = boundary_id_to_reduced_dim_.size() * dim;
				reduced_to_full_ = [this](const Eigen::VectorXd &reduced) {
					Eigen::VectorXd full;
					full.setZero(time_steps_ * boundary_ids_list_.size());
					for (int t = 0; t < time_steps_; ++t)
					{
						for (int b = 0; b < boundary_ids_list_.size(); ++b)
						{
							int boundary_id = boundary_ids_list_[b];
							if (boundary_id_to_reduced_dim_.count(boundary_id) == 0)
								continue;
							int k = b % dim_;
							int position = boundary_id_to_reduced_dim_.at(boundary_id);
							full(t * boundary_ids_list_.size() + b) = reduced(t * boundary_id_to_reduced_dim_.size() * dim_ + position * dim_ + k);
						}
					}
					return full;
				};
				full_to_reduced_ = [this](const Eigen::MatrixXd &full) {
					Eigen::VectorXd reduced;
					reduced.setZero(time_steps_ * boundary_id_to_reduced_dim_.size() * dim_);
					assert(full.size() == (boundary_ids_list_.size() * time_steps_));
					for (int t = 0; t < time_steps_; ++t)
						for (int b = 0; b < boundary_ids_list_.size(); ++b)
						{
							int boundary_id = boundary_ids_list_[b];
							if (boundary_id_to_reduced_dim_.count(boundary_id) == 0)
								continue;
							int k = b % dim_;
							int position = boundary_id_to_reduced_dim_.at(boundary_id);
							reduced(t * boundary_id_to_reduced_dim_.size() * dim_ + position * dim_ + k) = full(t * boundary_ids_list_.size() + b);
						}
					return reduced;
				};
				dfull_to_dreduced_ = [this](const Eigen::VectorXd &dfull) {
					Eigen::VectorXd dreduced;
					dreduced.setZero(time_steps_ * boundary_id_to_reduced_dim_.size() * dim_);
					assert(dfull.size() == (boundary_ids_list_.size() * time_steps_));
					for (int t = 0; t < time_steps_; ++t)
						for (int b = 0; b < boundary_ids_list_.size(); ++b)
						{
							int boundary_id = boundary_ids_list_[b];
							if (boundary_id_to_reduced_dim_.count(boundary_id) == 0)
								continue;
							int k = b % dim_;
							int position = boundary_id_to_reduced_dim_.at(boundary_id);
							dreduced(t * boundary_id_to_reduced_dim_.size() * dim_ + position * dim_ + k) += dfull(t * boundary_ids_list_.size() + b);
						}
					return dreduced;
				};
			}
		}

		void reduced_to_full(const Eigen::VectorXd &reduced, Eigen::MatrixXd &dirichlet_full) const
		{
			dirichlet_full = reduced_to_full_(reduced);
		}

		void full_to_reduced(const Eigen::MatrixXd &dirichlet_full, Eigen::VectorXd &reduced) const
		{
			reduced = full_to_reduced_(dirichlet_full);
		}

		void dfull_to_dreduced(const Eigen::VectorXd &dirichlet_full, const Eigen::VectorXd &d_dirichlet_full, Eigen::VectorXd &dreduced) const
		{
			dreduced = dfull_to_dreduced_(d_dirichlet_full);
		}

	private:
		const std::vector<int> boundary_ids_list_;
		const std::map<int, int> boundary_id_to_reduced_dim_;
		const int dim_;
		const int time_steps_;

		// Must define the inverse function with Eigen types, differentiability is not needed.
		std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> reduced_to_full_;

		// For differentiability, either define the function with differentiable types.
		// std::function<DiffVector(const DiffMatrix &)> full_to_reduced_diff_;
		// Or explicitly define the function and gradient with Eigen types.
		std::function<Eigen::VectorXd(const Eigen::MatrixXd &)> full_to_reduced_;
		std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> dfull_to_dreduced_;
	};
} // namespace polyfem