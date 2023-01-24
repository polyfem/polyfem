#pragma once

#include "Constraints.hpp"
#include <regex>

namespace polyfem
{
	class ControlConstraints : public Constraints
	{
	public:

		ControlConstraints(const json &constraint_params, const int time_steps, const int dim, const std::vector<int> &boundary_ids_list, const std::map<int, int> &boundary_id_to_reduced_dim, const Eigen::VectorXd &boundary_nodes_rest)
			: Constraints(constraint_params, boundary_ids_list.size(), 0), time_steps_(time_steps), dim_(dim), boundary_ids_list_(boundary_ids_list), boundary_id_to_reduced_dim_(boundary_id_to_reduced_dim), boundary_nodes_rest_(boundary_nodes_rest)
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
				int k = b % dim_;
				int pos = boundary_id_to_reduced_dim_.at(boundary_ids_list_[b]);
				initial_boundary_barycenter_(pos, k) += boundary_nodes_rest_(b) / reduced_dim_to_full_dim_multiplicity_.at(pos);
			}
			initial_boundary_nodes_relative_position_.setZero(boundary_ids_list_.size(), dim_);
			for (int b = 0; b < boundary_ids_list_.size(); ++b)
			{
				if (boundary_id_to_reduced_dim_.count(boundary_ids_list_[b]) == 0)
					continue;
				int k = b % dim_;
				if (k != 0)
					continue;
				int pos = boundary_id_to_reduced_dim_.at(boundary_ids_list_.at(b));
				Eigen::VectorXd rel_pos(dim_);
				for (int i = 0; i < dim_; ++i)
					rel_pos(i) = boundary_nodes_rest_(b + i);
				rel_pos -= initial_boundary_barycenter_.row(pos);
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
							if (dim_ == 2)
							{
								double theta = reduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 2);
								double dtheta = 0;
								if (k == 0)
								{
									dtheta += -(boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 0)) * std::sin(theta);
									dtheta += -(boundary_nodes_rest_(b + 1) - initial_boundary_barycenter_(position, 1)) * std::cos(theta);
								}
								else if (k == 1)
								{
									dtheta += (boundary_nodes_rest_(b - 1) - initial_boundary_barycenter_(position, 0)) * std::cos(theta);
									dtheta += -(boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 1)) * std::sin(theta);
								}
								dreduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 2) += dtheta * dfull(t * boundary_ids_list_.size() + b);
							}
							else if (dim_ == 3)
							{
								double phi = reduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 3);
								double theta = reduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 4);
								double psi = reduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 5);
								double dphi = 0;
								double dtheta = 0;
								double dpsi = 0;
								if (k == 0)
								{
									dphi += (boundary_nodes_rest_(b + 1) - initial_boundary_barycenter_(position, 1)) * (std::sin(phi) * std::sin(psi) + std::cos(phi) * std::sin(theta) * std::cos(psi));
									dphi += (boundary_nodes_rest_(b + 2) - initial_boundary_barycenter_(position, 2)) * (std::cos(phi) * std::sin(psi) - std::sin(phi) * std::sin(theta) * std::cos(psi));
									dtheta += -(boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 0)) * std::sin(theta) * std::cos(psi);
									dtheta += (boundary_nodes_rest_(b + 1) - initial_boundary_barycenter_(position, 1)) * std::sin(phi) * std::cos(theta) * std::cos(psi);
									dtheta += (boundary_nodes_rest_(b + 2) - initial_boundary_barycenter_(position, 2)) * std::cos(phi) * std::cos(theta) * std::cos(psi);
									dpsi += -(boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 0)) * std::cos(theta) * std::sin(psi);
									dpsi += (boundary_nodes_rest_(b + 1) - initial_boundary_barycenter_(position, 1)) * (-std::cos(phi) * std::sin(psi) - std::sin(phi) * std::sin(theta) * std::sin(psi));
									dpsi += (boundary_nodes_rest_(b + 2) - initial_boundary_barycenter_(position, 2)) * (std::sin(phi) * std::cos(psi) - std::cos(phi) * std::sin(theta) * std::sin(psi));
								}
								else if (k == 1)
								{
									dphi += (boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 1)) * (-std::sin(phi) * std::cos(psi) + std::cos(phi) * std::sin(theta) * std::sin(psi));
									dphi += (boundary_nodes_rest_(b + 1) - initial_boundary_barycenter_(position, 2)) * (-std::cos(phi) * std::cos(psi) - std::sin(phi) * std::sin(theta) * std::sin(psi));
									dtheta += -(boundary_nodes_rest_(b - 1) - initial_boundary_barycenter_(position, 0)) * std::sin(theta) * std::sin(psi);
									dtheta += (boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 1)) * std::sin(phi) * std::cos(theta) * std::sin(psi);
									dtheta += (boundary_nodes_rest_(b + 1) - initial_boundary_barycenter_(position, 2)) * std::cos(phi) * std::cos(theta) * std::sin(psi);
									dpsi += (boundary_nodes_rest_(b - 1) - initial_boundary_barycenter_(position, 0)) * std::cos(theta) * std::cos(psi);
									dpsi += (boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 1)) * (-std::cos(phi) * std::sin(psi) + std::sin(phi) * std::sin(theta) * std::cos(psi));
									dpsi += (boundary_nodes_rest_(b + 1) - initial_boundary_barycenter_(position, 2)) * (std::sin(phi) * std::sin(psi) + std::cos(phi) * std::sin(theta) * std::cos(psi));
								}
								else if (k == 2)
								{
									dphi += (boundary_nodes_rest_(b - 1) - initial_boundary_barycenter_(position, 1)) * std::cos(phi) * std::cos(theta);
									dphi += -(boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 2)) * std::sin(phi) * std::cos(theta);
									dtheta += -(boundary_nodes_rest_(b - 2) - initial_boundary_barycenter_(position, 0)) * std::cos(theta);
									dtheta += -(boundary_nodes_rest_(b - 1) - initial_boundary_barycenter_(position, 1)) * std::sin(phi) * std::sin(theta);
									dtheta += -(boundary_nodes_rest_(b) - initial_boundary_barycenter_(position, 2)) * std::cos(phi) * std::sin(theta);
								}
								dreduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 3) += dphi * dfull(t * boundary_ids_list_.size() + b);
								dreduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 4) += dtheta * dfull(t * boundary_ids_list_.size() + b);
								dreduced(t * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + position * (dim_ - 1) * 3 + 5) += dpsi * dfull(t * boundary_ids_list_.size() + b);
							}
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
				if (dim_ == 2)
					string_fn_vec_ = {[this](const int i, const Eigen::VectorXd &x) { return fmt::format("(x - {:16g}) * cos({:16g}) - (y - {:16g}) * sin({:16g}) + {:16g} + {:16g} - x", initial_boundary_barycenter_(i, 0), x(2), initial_boundary_barycenter_(i, 1), x(2), x(0), initial_boundary_barycenter_(i, 0)); },
									  [this](const int i, const Eigen::VectorXd &x) { return fmt::format("(x - {:16g}) * sin({:16g}) + (y - {:16g}) * cos({:16g}) + {:16g} + {:16g} - y", initial_boundary_barycenter_(i, 0), x(2), initial_boundary_barycenter_(i, 1), x(2), x(1), initial_boundary_barycenter_(i, 1)); }};
				else if (dim == 3)
					string_fn_vec_ = {[this](const int i, const Eigen::VectorXd &x) {
										  double phi = x(3);
										  double theta = x(4);
										  double psi = x(5);
										  return fmt::format("(x - {:16g}) * cos({:16g}) * cos({:16g}) + (y - {:16g}) * (-cos({:16g}) * sin({:16g}) + sin({:16g}) * sin({:16g}) * cos({:16g})) + (z - {:16g}) * (sin({:16g}) * sin({:16g}) + cos({:16g}) * sin({:16g}) * cos({:16g})) + {:16g} + {:16g} - x",
															 initial_boundary_barycenter_(i, 0), theta, psi, initial_boundary_barycenter_(i, 1), phi, psi, phi, theta, psi, initial_boundary_barycenter_(i, 2), phi, psi, phi, theta, psi, x(0), initial_boundary_barycenter_(i, 0));
									  },
									  [this](const int i, const Eigen::VectorXd &x) {
										  double phi = x(3);
										  double theta = x(4);
										  double psi = x(5);
										  return fmt::format("(x - {:16g}) * cos({:16g}) * sin({:16g}) + (y - {:16g}) * (cos({:16g}) * cos({:16g}) + sin({:16g}) * sin({:16g}) * sin({:16g})) + (z - {:16g}) * (-sin({:16g}) * cos({:16g}) + cos({:16g}) * sin({:16g}) * sin({:16g})) + {:16g} + {:16g} - y",
															 initial_boundary_barycenter_(i, 0), theta, psi, initial_boundary_barycenter_(i, 1), phi, psi, phi, theta, psi, initial_boundary_barycenter_(i, 2), phi, psi, phi, theta, psi, x(1), initial_boundary_barycenter_(i, 1));
									  },
									  [this](const int i, const Eigen::VectorXd &x) {
										  double phi = x(3);
										  double theta = x(4);
										  double psi = x(5);
										  return fmt::format("-(x - {:16g}) * sin({:16g}) + (y - {:16g}) * sin({:16g}) * cos({:16g}) + (z - {:16g}) * cos({:16g}) * cos({:16g}) + {:16g} + {:16g} - z",
															 initial_boundary_barycenter_(i, 0), theta, initial_boundary_barycenter_(i, 1), phi, theta, initial_boundary_barycenter_(i, 2), phi, theta, x(2), initial_boundary_barycenter_(i, 2));
									  }};
				constraint_to_string = [this](const Eigen::VectorXd &reduced) {
					std::regex whitespace(" ");
					std::map<int, std::vector<std::vector<std::string>>> constraint_string;
					for (auto kv : boundary_id_to_reduced_dim_)
					{
						std::vector<std::vector<std::string>> constraint_string_boundary;
						for (int k = 0; k < dim_; ++k)
						{
							std::vector<std::string> constraint_string_dim;
							for (int i = 0; i < time_steps_; ++i)
							{
								std::string bc_string = string_fn_vec_[k](kv.second, reduced.segment(i * boundary_id_to_reduced_dim_.size() * (dim_ - 1) * 3 + kv.second * (dim_ - 1) * 3, (dim_ - 1) * 3));
								constraint_string_dim.push_back(std::regex_replace(bc_string, whitespace, ""));
							}
							constraint_string_boundary.push_back(constraint_string_dim);
						}
						constraint_string[kv.first] = constraint_string_boundary;
					}
					return constraint_string;
				};
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
		Eigen::VectorXd boundary_nodes_rest_;
		const int dim_;
		const int time_steps_;

		std::vector<std::function<std::string(const int, const Eigen::VectorXd &)>> string_fn_vec_;

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