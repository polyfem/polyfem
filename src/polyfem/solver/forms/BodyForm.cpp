#include "BodyForm.hpp"

namespace polyfem
{
	namespace solver
	{
		BodyForm::BodyForm(const State &state, const assembler::RhsAssembler &rhs_assembler)
			: state_(state), rhs_assembler_(rhs_assembler), assembler_(state.assembler)
		{
			rhs_computed_ = false;
			t_ = 0;
		}

		double BodyForm::value(const Eigen::VectorXd &x)
		{
			const auto &gbases = state_.iso_parametric() ? state_.bases : state_.geom_bases;
			return rhs_assembler_.compute_energy(x, state_.local_neumann_boundary, state_.density, state_.n_boundary_samples(), t_);
		}

		void BodyForm::first_derivative(const Eigen::VectorXd &, Eigen::VectorXd &gradv)
		{
			//REMEMBER -!!!!!
			gradv = -current_rhs();
		}

		void BodyForm::update_quantities(const double t, const Eigen::VectorXd &x)
		{
			rhs_computed_ = false;
			this->t_ = t;
		}

		const Eigen::MatrixXd &BodyForm::current_rhs()
		{
			if (!rhs_computed_)
			{
				rhs_assembler_.compute_energy_grad(state_.local_boundary, state_.boundary_nodes, state_.density, state_.n_boundary_samples(), state_.local_neumann_boundary, state_.rhs, t_, current_rhs_);
				rhs_computed_ = true;

				if (assembler_.is_mixed(state_.formulation()))
				{
					const int prev_size = current_rhs_.size();
					//TODO check me
					// if (prev_size < full_size)
					current_rhs_.conservativeResize(prev_size + state_.n_pressure_bases, current_rhs_.cols());
					current_rhs_.block(prev_size, 0, state_.n_pressure_bases, current_rhs_.cols()).setZero();
				}

				rhs_assembler_.set_bc(std::vector<mesh::LocalBoundary>(), std::vector<int>(), state_.n_boundary_samples(), state_.local_neumann_boundary, current_rhs_, t_);

				//TODO: Check me
				//if (reduced_size != full_size)
				rhs_assembler_.set_bc(state_.local_boundary, state_.boundary_nodes, state_.n_boundary_samples(), std::vector<mesh::LocalBoundary>(), current_rhs_, t_);
			}

			return current_rhs_;
		}

	} // namespace solver
} // namespace polyfem
