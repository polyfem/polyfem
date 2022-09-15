#pragma once

#include <polyfem/Common.hpp>

// Line search methods
#include "line_search/LineSearch.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

#include <cppoptlib/solver/isolver.h>

namespace cppoptlib
{
	template <typename ProblemType /*, int Ord*/>
	class NonlinearSolver : public ISolver<ProblemType, /*Ord=*/-1>
	{
	public:
		using Superclass = ISolver<ProblemType, /*Ord=*/-1>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		NonlinearSolver(const polyfem::json &solver_params);

		virtual std::string name() const = 0;

		void set_line_search(const std::string &line_search_name);

		void minimize(ProblemType &objFunc, TVector &x);

		double line_search(const TVector &x, const TVector &delta_x, ProblemType &objFunc);

		void get_info(polyfem::json &params) { params = solver_info; }

		enum class ErrorCode
		{
			NanEncountered = -10,
			StepTooSmall = -1,
			Success = 0,
		};
		ErrorCode error_code() const { return m_error_code; }

	protected:
		// Reset the solver at the start of a minimization
		virtual void reset(const ProblemType &objFunc, const TVector &x);

		// Compute the search/update direction
		virtual bool compute_update_direction(
			ProblemType &objFunc,
			const TVector &x_vec,
			const TVector &grad,
			TVector &direction) = 0;

		virtual int default_descent_strategy() = 0;
		virtual void increase_descent_strategy() = 0;

		virtual std::string descent_strategy_name(int descent_strategy) const = 0;
		virtual std::string descent_strategy_name() const { return descent_strategy_name(this->descent_strategy); };

		virtual void update_solver_info();
		void reset_times();
		void log_times();

		std::shared_ptr<polyfem::solver::line_search::LineSearch<ProblemType>> m_line_search;

		const polyfem::json solver_params;
		ErrorCode m_error_code;
		bool use_gradient_norm;
		bool normalize_gradient;
		double use_grad_norm_tol;
		double first_grad_norm_tol;

		int descent_strategy; // 0, newton, 1 spd, 2 gradiant

		polyfem::json solver_info;

		double total_time;
		double grad_time;
		double assembly_time;
		double inverting_time;
		double line_search_time;
		double constraint_set_update_time;
		double obj_fun_time;
	};
} // namespace cppoptlib

#include "NonlinearSolver.tpp"
