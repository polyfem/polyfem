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
		BoundarySmoothingForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const bool scale_invariant, const int power) : AdjointForm(variable_to_simulations),
																																													state_(state),
																																													scale_invariant_(scale_invariant),
																																													power_(power) { init_form(); }

		double value_unweighted(const Eigen::VectorXd &x) const override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	private:
		void init_form();

		const State &state_;
		const bool scale_invariant_;
		const int power_; // only if scale_invariant_ is true
		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
	};

} // namespace polyfem::solver