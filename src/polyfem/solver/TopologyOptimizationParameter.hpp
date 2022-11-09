#pragma once

#include "Parameter.hpp"

namespace polyfem
{
	class TopologyOptimizationParameter : public Parameter
	{
	public:
		TopologyOptimizationParameter(std::vector<std::shared_ptr<State>> states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override;

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		Eigen::MatrixXd map(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const override;
		
		Eigen::VectorXd force_inequality_constraint(const Eigen::VectorXd &x0, const Eigen::VectorXd &dx) override;
		int n_inequality_constraints() override;
		double inequality_constraint_val(const Eigen::VectorXd &x, const int index) override;
		Eigen::VectorXd inequality_constraint_grad(const Eigen::VectorXd &x, const int index) override;

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const override;

		bool pre_solve(const Eigen::VectorXd &newX) override;

		void build_filter(const json &filter_args);
		Eigen::VectorXd apply_filter(const Eigen::VectorXd &x) const;
		Eigen::VectorXd apply_filter_to_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const;

	private:
		double max_change = 1;
		double min_density = 0;
		double max_density = 1;

		Eigen::VectorXd initial_density_;

		// E = E0 * pow(filter(x), density_power_)
		double density_power_;
		double lambda0, mu0;

		double min_mass = 0;
		double max_mass = 1;
		bool has_mass_constraint = false;

		json topo_params;

		bool has_filter;
		Eigen::SparseMatrix<double> tt_radius_adjacency;
		Eigen::VectorXd tt_radius_adjacency_row_sum;
	};
} // namespace polyfem