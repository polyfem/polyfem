#include "BodyForm.hpp"

namespace polyfem::solver
{
	BodyForm::BodyForm(const State &state, const assembler::RhsAssembler &rhs_assembler, const bool apply_DBC)
		: state_(state), rhs_assembler_(rhs_assembler), apply_DBC_(apply_DBC)
	{
		is_formulation_mixed_ = state.assembler.is_mixed(state.formulation());

		ndof_ = state.n_bases * state.mesh->dimension();
		if (is_formulation_mixed_)
			ndof_ += state.n_pressure_bases; // Pressure is a scalar

		t_ = 0;
		update_current_rhs();
	}

	double BodyForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return rhs_assembler_.compute_energy(x, state_.local_neumann_boundary, state_.density, state_.n_boundary_samples(), t_);
	}

	void BodyForm::first_derivative_unweighted(const Eigen::VectorXd &, Eigen::VectorXd &gradv) const
	{
		// REMEMBER -!!!!!
		gradv = -current_rhs_;
	}

	void BodyForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		this->t_ = t;
		update_current_rhs();
	}

	void BodyForm::update_current_rhs()
	{
		rhs_assembler_.compute_energy_grad(
			state_.local_boundary, state_.boundary_nodes, state_.density,
			state_.n_boundary_samples(), state_.local_neumann_boundary,
			state_.rhs, t_, current_rhs_);

		if (is_formulation_mixed_ && current_rhs_.size() < ndof_)
		{
			current_rhs_.conservativeResize(
				current_rhs_.rows() + state_.n_pressure_bases, current_rhs_.cols());
			current_rhs_.bottomRows(state_.n_pressure_bases).setZero();
		}

		// Apply Neumann boundary conditions
		rhs_assembler_.set_bc(
			std::vector<mesh::LocalBoundary>(), std::vector<int>(),
			state_.n_boundary_samples(), state_.local_neumann_boundary,
			current_rhs_, t_);

		// Apply Dirichlet boundary conditions
		if (apply_DBC_)
			rhs_assembler_.set_bc(
				state_.local_boundary, state_.boundary_nodes,
				state_.n_boundary_samples(), std::vector<mesh::LocalBoundary>(),
				current_rhs_, t_);
	}
} // namespace polyfem::solver
