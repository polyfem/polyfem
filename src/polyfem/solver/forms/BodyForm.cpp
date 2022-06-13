#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	namespace solver
	{
		BodyForm::BodyForm()
		{
		}

		double BodyForm::value(const Eigen::VectorXd &x)
		{
			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			return rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.density, state.n_boundary_samples(), t);
		}

		void BodyForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			//REMEMBER -!!!!!
			return current_rhs();
		}

		void BodyForm::update_quantities(const double t, const Eigen::VectorXd &x)
		{
			rhs_computed = false;
			this->t = t;
		}

		const Eigen::MatrixXd &BodyForm::current_rhs()
		{
			if (!rhs_computed)
			{
				rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.density, state.n_boundary_samples(), state.local_neumann_boundary, state.rhs, t, _current_rhs);
				rhs_computed = true;

				if (assembler.is_mixed(state.formulation()))
				{
					const int prev_size = _current_rhs.size();
					if (prev_size < full_size)
					{
						_current_rhs.conservativeResize(prev_size + state.n_pressure_bases, _current_rhs.cols());
						_current_rhs.block(prev_size, 0, state.n_pressure_bases, _current_rhs.cols()).setZero();
					}
				}
				assert(_current_rhs.size() == full_size);
				rhs_assembler.set_bc(std::vector<mesh::LocalBoundary>(), std::vector<int>(), state.n_boundary_samples(), state.local_neumann_boundary, _current_rhs, t);

				if (!ignore_inertia && is_time_dependent)
				{
					_current_rhs *= time_integrator()->acceleration_scaling();
					_current_rhs += state.mass * time_integrator()->x_tilde();
				}

				if (reduced_size != full_size)
				{
					// rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), state.local_neumann_boundary, _current_rhs, t);
					rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), std::vector<mesh::LocalBoundary>(), _current_rhs, t);
				}
			}

			return _current_rhs;
		}

	} // namespace solver
} // namespace polyfem
