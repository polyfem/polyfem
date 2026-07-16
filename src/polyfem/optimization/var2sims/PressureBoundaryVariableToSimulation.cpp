#include <polyfem/optimization/var2sims/PressureBoundaryVariableToSimulation.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/optimization/var2sims/ActiveSelectionUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

#include <cassert>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

namespace polyfem::solver
{

	PressureBoundaryVariableToSimulation::PressureBoundaryVariableToSimulation(
		StatePtrs states,
		DiffCachePtrs diff_caches,
		CompositeParametrization parametrizations,
		Eigen::VectorXi active_boundary_ids,
		Eigen::VectorXi active_time_slices)
		: is_transient_(states[0]->problem->is_time_dependent()),
		  time_steps_(0),
		  states_(std::move(states)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations)),
		  active_boundary_ids_(std::move(active_boundary_ids)),
		  active_time_slices_(std::move(active_time_slices))
	{
		assert(!states_.empty());
		assert(states_.size() == diff_caches_.size());

		for (auto &s : states_)
		{
			if (s->problem->is_time_dependent() != is_transient_)
			{
				log_and_throw_adjoint_error("Fail to construct pressure boundary variable to simulation. Reason: inconsistent transient/static states.");
			}
		}

		// time_step field might not be populated for static problem.
		if (is_transient_)
		{
			time_steps_ = states_[0]->args["time"]["time_steps"].get<int>();
		}

		// Expand implicit all-active boundary id selection (keep JSON order; no sort/unique pass).
		if (active_boundary_ids_.size() == 0)
		{
			json boundary_json = states_[0]->args["boundary_conditions"]["pressure_boundary"];
			std::vector<int> tmp;
			for (const json &bc : utils::json_as_array(boundary_json))
			{
				tmp.push_back(bc["id"].get<int>());
			}

			if (tmp.empty())
			{
				log_and_throw_adjoint_error("Fail to construct pressure boundary variable to simulation. Reason: No pressure boundary");
			}

			active_boundary_ids_ = Eigen::Map<Eigen::VectorXi>(tmp.data(), tmp.size());
		}

		// Expand implicit all-active time slice selection (transient only).
		if (is_transient_ && active_time_slices_.size() == 0)
		{
			active_time_slices_ = Eigen::VectorXi::LinSpaced(time_steps_, 0, time_steps_ - 1);
		}

		// Validate expanded selections against every state.
		std::string reason;
		if (!is_active_pressure_boundary_ids_valid(active_boundary_ids_, states_, reason))
		{
			log_and_throw_adjoint_error("Fail to construct pressure boundary variable to simulation. Reason: {}", reason);
		}
		if (is_transient_ && !is_active_time_slices_valid(active_time_slices_, states_, reason))
		{
			log_and_throw_adjoint_error("Fail to construct pressure boundary variable to simulation. Reason: {}", reason);
		}
	}

	std::string PressureBoundaryVariableToSimulation::name() const
	{
		return "pressure";
	}

	ParameterType PressureBoundaryVariableToSimulation::parameter_type() const
	{
		return ParameterType::PressureBC;
	}

	bool PressureBoundaryVariableToSimulation::affect_state(const legacy::State &target) const
	{
		for (auto &s : states_)
		{
			if (s.get() == &target)
			{
				return true;
			}
		}
		return false;
	}

	void PressureBoundaryVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		for (auto &s : states_)
		{
			auto tensor_problem = std::dynamic_pointer_cast<polyfem::assembler::GenericTensorProblem>(s->problem);
			if (!tensor_problem)
			{
				log_and_throw_adjoint_error("Only tensor problems are supported.");
			}

			if (is_transient_)
			{
				for (int ti = 0; ti < active_time_slices_.size(); ++ti)
				{
					int time_step = active_time_slices_(ti) + 1;
					for (int bi = 0; bi < active_boundary_ids_.size(); ++bi)
					{
						int boundary_id = active_boundary_ids_(bi);
						tensor_problem->update_pressure_boundary(boundary_id, time_step, y(ti * active_boundary_ids_.size() + bi));
					}
				}
			}
			else
			{
				for (int bi = 0; bi < active_boundary_ids_.size(); ++bi)
				{
					int boundary_id = active_boundary_ids_(bi);
					tensor_problem->update_pressure_boundary(boundary_id, /*time_step=*/1, y(bi));
				}
			}
		}
	}

	void PressureBoundaryVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == para_out_dof());
		state_variables = parametrization_.eval(x);
	}

	Eigen::VectorXd PressureBoundaryVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term = Eigen::VectorXd::Zero(para_out_dof());
		int bnum = active_boundary_ids_.size();

		// AdjointTool helps take std::vector<int> instead of Eigen::Vector.
		// Create temp to workaround this.
		std::vector<int> tmp(active_boundary_ids_.data(), active_boundary_ids_.data() + bnum);

		for (int si = 0; si < states_.size(); ++si)
		{
			auto &state = states_[si];
			auto &diff_cache = diff_caches_[si];

			if (is_transient_)
			{
				Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
				Eigen::MatrixXd adjoint_nu = get_adjoint_mat(*state, *diff_cache, 1);

				Eigen::VectorXd cur_term;
				AdjointTools::dJ_pressure_transient_adjoint_term(*state, *diff_cache, tmp, adjoint_nu, adjoint_p, cur_term);

				assert(cur_term.size() == time_steps_ * bnum);
				for (int ti = 0; ti < active_time_slices_.size(); ++ti)
				{
					int slice = active_time_slices_(ti);
					term.segment(ti * bnum, bnum) += cur_term.segment(slice * bnum, bnum);
				}
			}
			else
			{
				Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
				Eigen::VectorXd cur_term;
				AdjointTools::dJ_pressure_static_adjoint_term(*state, tmp, diff_cache->u(0), adjoint_p, cur_term);
				assert(cur_term.size() == bnum);
				term += cur_term;
			}
		}

		assert(term.size() == para_out_dof());
		return parametrization_.apply_jacobian(term, x);
	}

	int PressureBoundaryVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd PressureBoundaryVariableToSimulation::inverse_eval() const
	{
		Eigen::VectorXd y = Eigen::VectorXd::Zero(para_out_dof());
		int bnum = active_boundary_ids_.size();

		json boundary_json = states_[0]->args["boundary_conditions"]["pressure_boundary"];
		std::vector<json> boundaries = utils::json_as_array(boundary_json);

		for (int bi = 0; bi < bnum; ++bi)
		{
			int boundary_id = active_boundary_ids_(bi);

			auto pred = [boundary_id](const json &bc) { return bc["id"].get<int>() == boundary_id; };
			auto iter = std::find_if(boundaries.begin(), boundaries.end(), pred);
			if (iter == boundaries.end())
			{
				logger().warn("Cannot find pressure boundary id {} in JSON; falling back to zero.", boundary_id);
				continue;
			}

			const json &value = (*iter)["value"];
			if (is_transient_)
			{
				Eigen::VectorXd pressures;
				try
				{
					pressures = value;
				}
				catch (std::exception &err)
				{
				}

				int required = time_steps_ + 1;
				if (pressures.size() != required)
				{
					logger().warn("Unsupported initial value spec for pressure boundary id {}; falling back to zero.", boundary_id);
					pressures = Eigen::VectorXd::Zero(required);
				}

				for (int ti = 0; ti < active_time_slices_.size(); ++ti)
				{
					int slice = active_time_slices_(ti);
					y(ti * bnum + bi) = pressures(slice + 1);
				}
			}
			else
			{
				if (value.is_number())
				{
					y(bi) = value.get<double>();
				}
				else
				{
					logger().warn("Unsupported initial value spec for pressure boundary id {}; falling back to zero.", boundary_id);
				}
			}
		}

		return parametrization_.inverse_eval(y);
	}

	Eigen::VectorXd PressureBoundaryVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &, const Eigen::VectorXd &) const
	{
		// Not implemented because there's no user
		log_and_throw_adjoint_error("apply_parametrization_jacobian is not implemented in {} variable to simulation.", name());
	}

	int PressureBoundaryVariableToSimulation::para_out_dof() const
	{
		return is_transient_ ? (active_time_slices_.size() * active_boundary_ids_.size()) : active_boundary_ids_.size();
	}

} // namespace polyfem::solver
