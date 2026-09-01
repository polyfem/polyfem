#include <polyfem/optimization/var2sims/InitialConditionVariableToSimulation.hpp>

#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/VarFormDiff.hpp>
#include <polyfem/optimization/var2sims/ActiveSelectionUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

#include <cassert>
#include <string>

namespace polyfem::solver
{

	InitialConditionVariableToSimulation::InitialConditionVariableToSimulation(VarFormPtrs varforms,
																			   DiffCachePtrs diff_caches,
																			   CompositeParametrization parametrizations,
																			   Eigen::VectorXi active_dofs)
		: dof_num_(varforms[0]->primary_space().ndof()),
		  varforms_(std::move(varforms)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations)),
		  active_dofs_(std::move(active_dofs))
	{
		assert(!varforms_.empty());
		assert(varforms_.size() == diff_caches_.size());

		for (auto &varform : varforms_)
		{
			if (!varform->get_problem().is_time_dependent())
			{
				log_and_throw_adjoint_error("Fail to construct initial condition variable to simulation. Reason: Static problem not supported.");
			}
		}

		// Validate active selection.
		std::string reason;
		if (!is_active_dofs_valid(active_dofs_, varforms_, reason))
		{
			log_and_throw_adjoint_error("Fail to construct initial condition variable to simulation. Reason: {}", reason);
		}

		// Expand implicit all active selection.
		if (active_dofs_.size() == 0)
		{
			active_dofs_ = Eigen::VectorXi::LinSpaced(dof_num_, 0, dof_num_ - 1);
		}

		// Populate baseline initial-condition override for each varform.
		for (int i = 0; i < varforms_.size(); ++i)
		{
			Eigen::MatrixXd sol, vel;
			varforms_[i]->initial_solution(sol);
			varforms_[i]->initial_velocity(vel);

			// initial condition might return history of position and velocity.
			// Since we can't handle that, drop all condition from prev time steps.
			if (sol.cols() > 1)
			{
				sol.conservativeResize(Eigen::NoChange, 1);
			}
			if (vel.cols() > 1)
			{
				vel.conservativeResize(Eigen::NoChange, 1);
			}

			if (sol.rows() != dof_num_ || sol.cols() != 1)
			{
				log_and_throw_adjoint_error("Fail to construct initial condition variable to simulation. Reason: Invalid initial solution shape ({}, {}). Expect ({}, 1).",
											sol.rows(), sol.cols(), dof_num_);
			}
			if (vel.rows() != dof_num_ || vel.cols() != 1)
			{
				log_and_throw_adjoint_error("Fail to construct initial condition variable to simulation. Reason: Invalid initial velocity shape ({}, {}). Expect ({}, 1).",
											vel.rows(), vel.cols(), dof_num_);
			}

			diff_caches_[i]->initial_condition_override = varform::InitialConditionOverride{
				sol, vel, {}};
		}
	}

	std::string InitialConditionVariableToSimulation::name() const
	{
		return "initial";
	}

	ParameterType InitialConditionVariableToSimulation::parameter_type() const
	{
		return ParameterType::InitialCondition;
	}

	bool InitialConditionVariableToSimulation::affects_varform(const varform::DifferentiableVarForm &target) const
	{
		for (const auto &varform : varforms_)
		{
			if (varform.get() == &target)
			{
				return true;
			}
		}
		return false;
	}

	void InitialConditionVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		int active_num = active_dofs_.size();
		for (auto &dc : diff_caches_)
		{
			// Override should already be populated in the constructor.
			assert(dc->initial_condition_override && "Initial-condition optimization must initialize its override");
			auto &initial_condition_override = *dc->initial_condition_override;
			auto &sol = initial_condition_override.solution;
			auto &vel = initial_condition_override.velocity;
			assert(sol.rows() == dof_num_ && sol.cols() >= 1 && "Initial solution override must match the simulation DOFs");
			assert(vel.rows() == dof_num_ && vel.cols() >= 1 && "Initial velocity override must match the simulation DOFs");

			for (int i = 0; i < active_num; ++i)
			{
				sol(active_dofs_(i), 0) = y(i);
				vel(active_dofs_(i), 0) = y(active_num + i);
			}

			initial_condition_override.acceleration = {};
		}
	}

	void InitialConditionVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == 2 * dof_num_);

		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		int active_num = active_dofs_.size();
		for (int i = 0; i < active_num; ++i)
		{
			state_variables(active_dofs_(i)) = y(i);
			state_variables(dof_num_ + active_dofs_(i)) = y(active_num + i);
		}
	}

	Eigen::VectorXd InitialConditionVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (int i = 0; i < varforms_.size(); ++i)
		{
			auto &varform = varforms_[i];
			auto &diff_cache = diff_caches_[i];

			Eigen::MatrixXd adjoint_p = get_adjoint_mat(*varform, *diff_cache, 0);
			Eigen::MatrixXd adjoint_nu = get_adjoint_mat(*varform, *diff_cache, 1);
			AdjointTools::dJ_initial_condition_adjoint_term(*varform, adjoint_nu, adjoint_p, cur_term);

			if (term.size() != cur_term.size())
			{
				term = cur_term;
			}
			else
			{
				term += cur_term;
			}
		}

		assert(term.size() == 2 * dof_num_);

		int active_num = active_dofs_.size();
		Eigen::VectorXd active_term(para_out_dof());
		for (int j = 0; j < active_num; ++j)
		{
			active_term(j) = term(active_dofs_(j));
			active_term(active_num + j) = term(dof_num_ + active_dofs_(j));
		}

		assert(active_term.size() == para_out_dof());
		return parametrization_.apply_jacobian(active_term, x);
	}

	int InitialConditionVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd InitialConditionVariableToSimulation::inverse_eval() const
	{
		assert(diff_caches_[0]->initial_condition_override && "Initial-condition optimization must initialize its override");
		const auto &initial_condition_override = *diff_caches_[0]->initial_condition_override;
		const Eigen::MatrixXd &sol = initial_condition_override.solution;
		const Eigen::MatrixXd &vel = initial_condition_override.velocity;
		assert(sol.rows() == dof_num_ && sol.cols() >= 1 && "Initial solution override must match the simulation DOFs");
		assert(vel.rows() == dof_num_ && vel.cols() >= 1 && "Initial velocity override must match the simulation DOFs");

		int active_num = active_dofs_.size();
		Eigen::VectorXd y(para_out_dof());
		for (int j = 0; j < active_num; ++j)
		{
			y(j) = sol(active_dofs_(j), 0);
			y(active_num + j) = vel(active_dofs_(j), 0);
		}
		return parametrization_.inverse_eval(y);
	}

	Eigen::VectorXd InitialConditionVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &, const Eigen::VectorXd &) const
	{
		// Not implemented because there's no user
		log_and_throw_adjoint_error("apply_parametrization_jacobian is not implemented in {} variable to simulation.", name());
	}

	int InitialConditionVariableToSimulation::para_out_dof() const
	{
		return 2 * active_dofs_.size();
	}

} // namespace polyfem::solver
