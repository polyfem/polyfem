#include "BodyForm.hpp"

#include <vector>

namespace polyfem::solver
{
	BodyForm::BodyForm(const int ndof,
					   const int n_pressure_bases,
					   const std::vector<int> &boundary_nodes,
					   const std::vector<mesh::LocalBoundary> &local_boundary,
					   const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
					   const QuadratureOrders &n_boundary_samples,
					   const Eigen::MatrixXd &rhs,
					   const assembler::RhsAssembler &rhs_assembler,
					   const assembler::Density &density,
					   const bool is_formulation_mixed,
					   const bool is_time_dependent)
		: ndof_(ndof),
		  n_pressure_bases_(n_pressure_bases),
		  boundary_nodes_(boundary_nodes),
		  local_boundary_(local_boundary),
		  local_neumann_boundary_(local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_(rhs),
		  rhs_assembler_(rhs_assembler),
		  density_(density),
		  is_formulation_mixed_(is_formulation_mixed)
	{
		t_ = 0;
		if (!is_time_dependent)
			update_current_rhs(Eigen::VectorXd());
	}

	double BodyForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return rhs_assembler_.compute_energy(x, x_prev_, local_neumann_boundary_, density_, n_boundary_samples_, t_);
	}

	void BodyForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		// REMEMBER -!!!!!
		gradv = -current_rhs_;
	}

	void BodyForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian.resize(x.size(), x.size());
	}

	void BodyForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		this->t_ = t;
		this->x_prev_ = x;
		update_current_rhs(x);
	}

	void BodyForm::update_current_rhs(const Eigen::VectorXd &x)
	{
		rhs_assembler_.compute_energy_grad(
			local_boundary_, boundary_nodes_, density_,
			n_boundary_samples_, local_neumann_boundary_,
			rhs_, t_, current_rhs_);

		if (is_formulation_mixed_ && current_rhs_.size() < ndof_)
		{
			current_rhs_.conservativeResize(
				current_rhs_.rows() + n_pressure_bases_, current_rhs_.cols());
			current_rhs_.bottomRows(n_pressure_bases_).setZero();
		}

		// Apply Neumann boundary conditions
		rhs_assembler_.set_bc(
			std::vector<mesh::LocalBoundary>(), std::vector<int>(),
			n_boundary_samples_, local_neumann_boundary_,
			current_rhs_, x, t_);
	}

	void BodyForm::hessian_wrt_u_prev(const Eigen::VectorXd &u_prev, const double t, StiffnessMatrix &hessian) const
	{
		rhs_assembler_.compute_energy_hess(boundary_nodes_, n_boundary_samples_, local_neumann_boundary_, u_prev, t, false, hessian);
		hessian *= -1;
	}
} // namespace polyfem::solver
