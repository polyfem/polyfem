#include "SolverInfo.hpp"

namespace polyfem::solver
{
	void SolverInfo::set_line_search(
		std::shared_ptr<polyfem::solver::line_search::LineSearch<ProblemType>> line_search_ptr)
	{
		m_line_search = line_search;
		info["line_search"] = line_search->name();
	}

	void SolverInfo::clear()
	{
		total_time = 0;
		grad_time = 0;
		assembly_time = 0;
		inverting_time = 0;
		line_search_time = 0;
		obj_fun_time = 0;
		constraint_set_update_time = 0;
		if (m_line_search)
		{
			m_line_search->reset_times();
		}
	}

	void SolverInfo::update()
	{
		info["status"] = this->status();
		info["error_code"] = m_error_code;

		const auto &crit = this->criteria();
		info["iterations"] = crit.iterations;
		info["xDelta"] = crit.xDelta;
		info["fDelta"] = crit.fDelta;
		info["gradNorm"] = crit.gradNorm;
		info["condition"] = crit.condition;
		info["use_gradient_norm"] = use_gradient_norm;
		info["relative_gradient"] = normalize_gradient;

		double per_iteration = crit.iterations ? crit.iterations : 1;

		info["total_time"] = total_time;
		info["time_grad"] = grad_time / per_iteration;
		info["time_assembly"] = assembly_time / per_iteration;
		info["time_inverting"] = inverting_time / per_iteration;
		info["time_line_search"] = line_search_time / per_iteration;
		info["time_constraint_set_update"] = constraint_set_update_time / per_iteration;
		info["time_obj_fun"] = obj_fun_time / per_iteration;

		if (m_line_search)
		{
			info["line_search_iterations"] = m_line_search->iterations;

			info["time_checking_for_nan_inf"] =
				m_line_search->checking_for_nan_inf_time / per_iteration;
			info["time_broad_phase_ccd"] =
				m_line_search->broad_phase_ccd_time / per_iteration;
			info["time_ccd"] = m_line_search->ccd_time / per_iteration;
			// Remove double counting
			info["time_classical_line_search"] =
				(m_line_search->classical_line_search_time
				 - m_line_search->constraint_set_update_time)
				/ per_iteration;
			info["time_line_search_constraint_set_update"] =
				m_line_search->constraint_set_update_time / per_iteration;
		}
	}

	void SolverInfo::log_times() const
	{
		polyfem::logger().debug(
			"[timing] grad {:.3g}s, assembly {:.3g}s, inverting {:.3g}s, "
			"line_search {:.3g}s, constraint_set_update {:.3g}s, "
			"obj_fun {:.3g}s, checking_for_nan_inf {:.3g}s, "
			"broad_phase_ccd {:.3g}s, ccd {:.3g}s, "
			"classical_line_search {:.3g}s",
			grad_time, assembly_time, inverting_time, line_search_time,
			constraint_set_update_time + (m_line_search ? m_line_search->constraint_set_update_time : 0),
			obj_fun_time, m_line_search ? m_line_search->checking_for_nan_inf_time : 0,
			m_line_search ? m_line_search->broad_phase_ccd_time : 0, m_line_search ? m_line_search->ccd_time : 0,
			m_line_search ? m_line_search->classical_line_search_time : 0);
	}

} // namespace polyfem::solver