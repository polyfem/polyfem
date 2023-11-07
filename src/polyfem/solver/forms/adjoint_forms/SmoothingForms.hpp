#pragma once

#include "AdjointForm.hpp"

namespace polyfem
{
	class State;
}

namespace polyfem::solver
{
	class BoundarySmoothingForm : public AdjointForm
	{
	public:
		BoundarySmoothingForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : AdjointForm(variable_to_simulations),
																																						 state_(state)
		{
			interested_boundary_ids_ = args["surface_selection"].get<std::vector<int>>();
			scale_invariant_ = args["scale_invariant"].get<bool>();
			power_ = args["power"].get<int>();
			init_form();
		}

		BoundarySmoothingForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const bool scale_invariant, const int power) : AdjointForm(variable_to_simulations),
																																													state_(state)
		{
			scale_invariant_ = scale_invariant;
			power_ = power;
			init_form();
		}

		double value_unweighted(const Eigen::VectorXd &x) const override;
		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		Eigen::MatrixXd compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const override;

	private:
		void init_form();

		const State &state_;
		std::vector<int> interested_boundary_ids_;
		bool scale_invariant_;
		int power_; // only if scale_invariant_ is true
		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
	};

} // namespace polyfem::solver