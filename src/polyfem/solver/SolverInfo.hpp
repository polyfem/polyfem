#pragma once

#include <polyfem/Common.hpp>

namespace polyfem::solver
{
	struct SolverInfo
	{
		json info;

		double total_time = 0;
		double grad_time = 0;
		double assembly_time = 0;
		double inverting_time = 0;
		double line_search_time = 0;
		double constraint_set_update_time = 0;
		double obj_fun_time = 0;

		void clear();
		void update();
		void log_times() const;
	};

} // namespace polyfem::solver