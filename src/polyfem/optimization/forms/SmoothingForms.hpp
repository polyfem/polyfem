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
		BoundarySmoothingForm(
			const VariableToSimulationGroup &variable_to_simulations,
			const State &state,
			const bool scale_invariant,
			const int power,
			const std::vector<int> &surface_selections,
			const std::vector<int> &active_dims);

		double value_unweighted(const Eigen::VectorXd &x) const override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	private:
		const State &state_;
		const bool scale_invariant_;
		const int power_; // only if scale_invariant_ is true
		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
		std::set<int> surface_ids_;
		std::vector<int> active_dims_;
	};
} // namespace polyfem::solver
