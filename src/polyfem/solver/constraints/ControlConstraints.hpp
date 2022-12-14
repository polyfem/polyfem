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

			for (auto kv : boundary_id_to_reduced_dim_)
			{
				int boundary_id = kv.first;
				int nodes = 0;
				for (int b = 0; b < boundary_ids_list_.size(); ++b)
				{
					if (boundary_ids_list_[b] == boundary_id)
						nodes++;
				}
				reduced_dim_to_full_dim_multiplicity_[kv.second] = nodes / dim_;
			}

			initial_boundary_barycenter_.setZero(boundary_id_to_reduced_dim_.size(), dim_);
			for (int b = 0; b < boundary_ids_list_.size(); ++b)
			{
				if (boundary_id_to_reduced_dim_.count(boundary_ids_list_[b]) == 0)
					continue;
				int pos = boundary_id_to_reduced_dim_.at(boundary_ids_list_[b]);
				initial_boundary_barycenter_.row(pos) += Eigen::MatrixXd::Zero(1, dim_) / reduced_dim_to_full_dim_multiplicity_.at(pos);
			}
			initial_boundary_nodes_relative_position_.setZero(boundary_ids_list_.size(), dim_);
			for (int b = 0; b < boundary_ids_list_.size(); ++b)
			{
				if (boundary_id_to_reduced_dim_.count(boundary_ids_list_[b]) == 0)
					continue;
				int pos = boundary_id_to_reduced_dim_.at(boundary_ids_list_.at(b));
				Eigen::VectorXd rel_pos = -initial_boundary_barycenter_.row(pos);
				initial_boundary_nodes_relative_position_(b, 0) = rel_pos.norm();
				initial_boundary_nodes_relative_position_(b, 1) = std::atan(rel_pos(1) - rel_pos(0));
				if (dim == 3)
					initial_boundary_nodes_relative_position_(b, 2) = std::atan(rel_pos(2) - rel_pos(1));
			}

			reduced_size_ = time_steps * boundary_id_to_reduced_dim_.size() * (dim - 1) * 3;
			if (restriction == "none")
			{
				dfull_to_dreduced_ = [this](const Eigen::VectorXd &reduced, const Eigen::VectorXd &dfull) {
					Eigen::VectorXd dreduced;
					dreduced.setZero(time_steps_ * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3);
					assert(dfull.size() == (boundary_ids_list_.size() * time_steps_));
					for (int t = 0; t < time_steps_; ++t)
						for (int b = 0; b < boundary_ids_list_.size(); ++b)
						{
							int boundary_id = boundary_ids_list_[b];
							if (boundary_id_to_reduced_dim_.count(boundary_id) == 0)
								continue;
							int k = b % dim_;
							int position = boundary_id_to_reduced_dim_.at(boundary_id);
							dreduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + k) += dfull(t * boundary_ids_list_.size() + b);
						}
					return dreduced;
				};
				// Needed to project the control smoothing grad to reduced parameters
				dreduced_to_dfull_ = [this](const Eigen::VectorXd &dreduced) {
					Eigen::VectorXd dfull;
					dfull.setZero(boundary_ids_list_.size());
					for (auto kv : boundary_id_to_reduced_dim_)
					{
						int boundary_id = kv.first;
						for (int b = 0; b < boundary_ids_list_.size(); ++b)
						{
							int k = b % dim_;
							if (boundary_ids_list_[b] == boundary_id)
								dfull(b) = dreduced(boundary_id_to_reduced_dim_.at(boundary_id) * (dim_ - 1) * 3 + k) / reduced_dim_to_full_dim_multiplicity_.at(boundary_id_to_reduced_dim_.at(boundary_id));
						}
					}
					return dfull;
				};
				constraint_to_string = [this](const Eigen::VectorXd &reduced) {
					std::map<int, std::vector<std::vector<std::string>>> constraint_string;
					for (auto kv : boundary_id_to_reduced_dim_)
					{
						std::vector<std::vector<std::string>> constraint_string_boundary;
						for (int k = 0; k < dim_; ++k)
						{
							std::vector<std::string> constraint_string_dim;
							for (int i = 0; i < time_steps_; ++i)
							{
								constraint_string_dim.push_back(fmt::format("{:16g}", reduced(i * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + kv.second * (dim_ - 1) * 3 + k)));
							}
							constraint_string_boundary.push_back(constraint_string_dim);
						}
						constraint_string[kv.first] = constraint_string_boundary;
					}
					return constraint_string;
				};
			}
			else if (restriction == "rigid")
			{
				if (dim == 2)
				{
					dfull_to_dreduced_ = [this](const Eigen::VectorXd &reduced, const Eigen::VectorXd &dfull) {
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
								dreduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + k) += dfull(t * boundary_ids_list_.size() + b);
								dreduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 2) -= initial_boundary_nodes_relative_position_(b, 2) * std::sin(reduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 2)) * dfull(t * boundary_ids_list_.size() + b);
							}
						return dreduced;
					};

					// Needed to project the control smoothing grad to reduced parameters
					dreduced_to_dfull_ = [this](const Eigen::VectorXd &dreduced) {
						Eigen::VectorXd dfull;
						dfull.setZero(boundary_ids_list_.size());
						for (auto kv : boundary_id_to_reduced_dim_)
						{
							int boundary_id = kv.first;
							for (int b = 0; b < boundary_ids_list_.size(); ++b)
							{
								int k = b % dim_;
								if (boundary_ids_list_[b] == boundary_id)
									dfull(b) = dreduced(boundary_id_to_reduced_dim_.at(boundary_id) * (dim_ - 1) * 3 + k) / reduced_dim_to_full_dim_multiplicity_.at(boundary_id_to_reduced_dim_.at(boundary_id));
							}
						}
						return dfull;
					};
					constraint_to_string = [this](const Eigen::VectorXd &reduced) {
						std::map<int, std::vector<std::vector<std::string>>> constraint_string;
						for (auto kv : boundary_id_to_reduced_dim_)
						{
							std::vector<std::vector<std::string>> constraint_string_boundary;
							for (int k = 0; k < dim_; ++k)
							{
								std::vector<std::string> constraint_string_dim;
								for (int i = 0; i < time_steps_; ++i)
								{
									constraint_string_dim.push_back(fmt::format("{:16g}", reduced(i * boundary_id_to_reduced_dim_.size() * dim_ + kv.second * dim_ + k)));
								}
								constraint_string_boundary.push_back(constraint_string_dim);
							}
							constraint_string[kv.first] = constraint_string_boundary;
						}
						return constraint_string;
					};
				}
				else
					assert(false);
			}
		}

		void dfull_to_dreduced(const Eigen::VectorXd &dirichlet_reduced, const Eigen::VectorXd &d_dirichlet_full, Eigen::VectorXd &dreduced) const
		{
			dreduced = dfull_to_dreduced_(dirichlet_reduced, d_dirichlet_full);
		}

		void dreduced_to_dfull_timestep(const Eigen::VectorXd &dreduced, Eigen::VectorXd &d_dirichlet_full) const
		{
			d_dirichlet_full = dreduced_to_dfull_(dreduced);
		}

		std::function<std::map<int, std::vector<std::vector<std::string>>>(const Eigen::VectorXd &)> constraint_to_string;

	private:
		const std::vector<int> boundary_ids_list_;
		const std::map<int, int> boundary_id_to_reduced_dim_;
		std::map<int, int> reduced_dim_to_full_dim_multiplicity_;
		Eigen::MatrixXd initial_boundary_barycenter_;
		Eigen::MatrixXd initial_boundary_nodes_relative_position_;
		const int dim_;
		const int time_steps_;

		// Must define the inverse function with Eigen types, differentiability is not needed.
		// std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> reduced_to_full_;

		// For differentiability, either define the function with differentiable types.
		// std::function<DiffVector(const DiffMatrix &)> full_to_reduced_diff_;
		// Or explicitly define the function and gradient with Eigen types.
		// std::function<Eigen::VectorXd(const Eigen::MatrixXd &)> full_to_reduced_;
		std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> dfull_to_dreduced_;
		std::function<Eigen::VectorXd(const Eigen::VectorXd &)> dreduced_to_dfull_;
	};
} // namespace polyfem