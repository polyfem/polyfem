#pragma once

#include <polyfem/Common.hpp>

// Line search methods
#include <polyfem/LineSearch.hpp>

#include <polyfem/Logger.hpp>
#include <polyfem/Timer.hpp>

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

		enum class ErrorCode
		{
			NanEncountered = -10,
			StepTooSmall = -1,
			Success = 0,
		};

		NonlinearSolver(const json &solver_params)
			: solver_params(solver_params)
		{
			auto criteria = this->criteria();
			criteria.fDelta = solver_params["fDelta"];
			criteria.gradNorm = solver_params["gradNorm"];
			criteria.iterations = solver_params["nl_iterations"];

			use_gradient_norm = solver_params["useGradNorm"];
			normalize_gradient = solver_params["relativeGradient"];
			use_grad_norm_tol = solver_params["use_grad_norm_tol"];
			this->setStopCriteria(criteria);

			setLineSearch("armijo");
		}

		virtual std::string name() const = 0;

		void setLineSearch(const std::string &line_search_name)
		{
			m_line_search = polyfem::LineSearch<ProblemType>::construct_line_search(line_search_name);
			solver_info["line_search"] = line_search_name;
		}

		void minimize(ProblemType &objFunc, TVector &x)
		{
			using namespace polyfem;

			// ---------------------------
			// Initialize the minimization
			// ---------------------------

			reset(objFunc, x); // place for children to initialize their fields

			TVector grad = TVector::Zero(x.rows());
			TVector delta_x = TVector::Zero(x.rows());

			// double factor = 1e-5;

			// Set these to nan to indicate they have not been computed yet
			double old_energy = std::nan("");

			double first_grad_norm = -1;

			Timer timer("non-linear solver", this->total_time);
			timer.start();

			m_line_search->use_grad_norm_tol = use_grad_norm_tol;

			do
			{
				{
					POLYFEM_SCOPED_TIMER("constraint set update", constraint_set_update_time);
					objFunc.solution_changed(x);
				}

				double energy;
				{
					POLYFEM_SCOPED_TIMER("compute objective function", obj_fun_time);
					energy = objFunc.value(x);
				}
				if (!std::isfinite(energy))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().error("[{}] f(x) is nan or inf; stopping", name());
					m_error_code = ErrorCode::NanEncountered;
					throw("f(x) is nan or inf; stopping");
					break;
				}

				{
					POLYFEM_SCOPED_TIMER("compute gradient", grad_time);
					objFunc.gradient(x, grad);
				}

				const double grad_norm = grad.norm();
				if (std::isnan(grad_norm))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().error("[{}] Gradient is nan; stopping", name());
					m_error_code = ErrorCode::NanEncountered;
					throw("Gradient is nan; stopping");
					break;
				}

				if (first_grad_norm < 0)
					first_grad_norm = grad_norm;

				// ------------------------
				// Compute update direction
				// ------------------------

				// Compute a Δx to update the variable
				if (!compute_update_direction(objFunc, x, grad, delta_x))
				{
					this->m_status = Status::Continue;
					continue;
				}

				const double delta_x_norm = delta_x.norm();
				if (std::isnan(delta_x_norm))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().error("[{}] Δx is nan; stopping", name());
					m_error_code = ErrorCode::NanEncountered;
					throw("Δx is nan; stopping");
					break;
				}

				if (!use_gradient_norm)
				{
					//TODO, we shold remove this
					// Use the maximum absolute displacement value divided by the timestep,
					// so the units are in velocity units.
					// TODO: Set this to the actual timestep
					double dt = 1;
					// TODO: Also divide by the world scale to make this criteria scale invariant.
					this->m_current.gradNorm = delta_x.template lpNorm<Eigen::Infinity>() / dt;
				}
				else
				{
					//if normalize_gradient, use relative to first norm
					this->m_current.gradNorm = grad_norm / (normalize_gradient ? first_grad_norm : 1);
				}
				this->m_current.fDelta = std::abs(old_energy - energy); // / std::abs(old_energy);

				this->m_status = checkConvergence(this->m_stop, this->m_current);

				old_energy = energy;

				// if (!use_gradient_norm && this->m_status != Status::Continue && this->m_status != Status::IterationLimit && grad_norm > 1e-2)
				// {
				// 	polyfem::logger().debug("[{}] converged with large gradient (||∇f||={}); continuing", name(), grad_norm);
				// 	this->m_status = Status::Continue;
				// }

				// // Converge on the first iteration iff the line search failed and the gradient is small
				// if ((descent_strategy == 2 || this->m_current.iterations > 0) && this->m_status != Status::Continue)
				// {
				// 	// converged
				// 	break;
				// }

				// ---------------
				// Variable update
				// ---------------

				// Perform a line_search to compute step scale
				double rate = line_search(x, delta_x, objFunc);
				if (std::isnan(rate))
				{
					// descent_strategy set by line_search upon failure
					if (this->m_status == Status::Continue)
						continue;
					else
						break;
				}

				x += rate * delta_x;

				// -----------
				// Post update
				// -----------

				descent_strategy = default_descent_strategy(); // Reset this for the next iterations

				const double step = (rate * delta_x).norm();
				// if (this->m_status == Status::Continue && step < 1e-10)
				// {
				// 	handle_small_step(step);
				// }

				if (objFunc.stop(x))
				{
					this->m_status = Status::UserDefined;
					m_error_code = ErrorCode::Success;
					polyfem::logger().debug("[{}] Objective decided to stop", name());
				}

				// if(rate >= 1 && next_hessian == this->m_current.iterations)
				//	next_hessian += 2;

				objFunc.post_step(this->m_current.iterations, x);

				polyfem::logger().debug(
					"[{}] iter={:} f={} ‖∇f‖={} ‖Δx‖={} Δx⋅∇f(x)={} g={} tol={} rate={} ‖step‖={}",
					name(), this->m_current.iterations, energy, grad_norm, delta_x_norm, delta_x.dot(grad),
					this->m_current.gradNorm, this->m_stop.gradNorm, rate, step);
				++this->m_current.iterations;
			} while (objFunc.callback(this->m_current, x) && (this->m_status == Status::Continue));

			timer.stop();

			// -----------
			// Log results
			// -----------

			std::string msg = "Finished";
			spdlog::level::level_enum level = spdlog::level::info;
			if (this->m_status == Status::IterationLimit)
			{
				msg = fmt::format("[{}] Reached iteration limit", name());
				polyfem::logger().error(msg);
				throw(msg);
				level = spdlog::level::err;
			}
			else if (this->m_current.iterations == 0)
			{
				msg = fmt::format("[{}] Unable to take a step", name());
				polyfem::logger().error(msg);
				throw(msg);
				level = this->m_status == Status::UserDefined ? spdlog::level::err : spdlog::level::warn;
			}
			else if (this->m_status == Status::UserDefined)
			{
				msg = fmt::format("[{}] Failed to find minimizer", name());
				polyfem::logger().error(msg);
				throw(msg);
				level = spdlog::level::err;
			}
			polyfem::logger().log(
				level, "[{}] {}, took {}s (niters={} f={} ||∇f||={} ||Δx||={} Δx⋅∇f(x)={} g={} tol={})",
				name(), msg, timer.getElapsedTimeInSec(), this->m_current.iterations, old_energy, grad.norm(), delta_x.norm(),
				delta_x.dot(grad), this->m_current.gradNorm, this->m_stop.gradNorm);

			log_times();
			update_solver_info();
		}

		double line_search(const TVector &x, const TVector &delta_x, ProblemType &objFunc)
		{
			POLYFEM_SCOPED_TIMER("line search", line_search_time);

			if (!m_line_search)
				return 1; // no linesearch

			double rate = m_line_search->line_search(x, delta_x, objFunc);

			if (std::isnan(rate) && descent_strategy < 2) // 2 is the max, grad descent
			{
				descent_strategy++;
				polyfem::logger().warn(
					"[{}] Line search failed; reverting to {}", name(), descent_strategy_name());
				this->m_status = Status::Continue; // Try the step again with gradient descent
			}
			else if (std::isnan(rate))
			{
				assert(descent_strategy == 2); // failed on gradient descent
				polyfem::logger().error("[{}] Line search failed on gradient descent; stopping", name());
				this->m_status = Status::UserDefined; // Line search failed on gradient descent, so quit!
				throw("Line search failed on gradient descent");
			}

			return rate;
		}

		void getInfo(json &params)
		{
			params = solver_info;
		}

		ErrorCode error_code() const { return m_error_code; }

	protected:
		// Reset the solver at the start of a minimization
		virtual void reset(const ProblemType &objFunc, const TVector &x)
		{
			this->m_current.reset();
			reset_times();
			m_error_code = ErrorCode::Success;
			descent_strategy = default_descent_strategy();
		}

		// Compute the search/update direction
		virtual bool compute_update_direction(ProblemType &objFunc, const TVector &x_vec, const TVector &grad, TVector &direction) = 0;

		// Special handling for small steps
		virtual void handle_small_step(double step) {}

	protected:
		const json solver_params;
		virtual int default_descent_strategy() = 0;
		virtual std::string descent_strategy_name(int descent_strategy) const = 0;
		virtual std::string descent_strategy_name() const { return descent_strategy_name(this->descent_strategy); };

		std::shared_ptr<polyfem::LineSearch<ProblemType>> m_line_search;

		int descent_strategy; //0, newton, 1 spd, 2 gradiant

		ErrorCode m_error_code;
		bool use_gradient_norm;
		bool normalize_gradient;
		double use_grad_norm_tol;

		json solver_info;

		double total_time;
		double grad_time;
		double assembly_time;
		double inverting_time;
		double line_search_time;
		double constraint_set_update_time;
		double obj_fun_time;

		void reset_times()
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

		virtual void update_solver_info()
		{
			solver_info["status"] = this->status();
			solver_info["error_code"] = m_error_code;

			const auto &crit = this->criteria();
			solver_info["iterations"] = crit.iterations;
			solver_info["xDelta"] = crit.xDelta;
			solver_info["fDelta"] = crit.fDelta;
			solver_info["gradNorm"] = crit.gradNorm;
			solver_info["condition"] = crit.condition;
			solver_info["use_gradient_norm"] = use_gradient_norm;
			solver_info["relative_gradient"] = normalize_gradient;

			double per_iteration = crit.iterations ? crit.iterations : 1;

			solver_info["total_time"] = total_time;
			solver_info["time_grad"] = grad_time / per_iteration;
			solver_info["time_assembly"] = assembly_time / per_iteration;
			solver_info["time_inverting"] = inverting_time / per_iteration;
			solver_info["time_line_search"] = line_search_time / per_iteration;
			solver_info["time_constraint_set_update"] = constraint_set_update_time / per_iteration;
			solver_info["time_obj_fun"] = obj_fun_time / per_iteration;

			if (m_line_search)
			{
				solver_info["line_search_iterations"] = m_line_search->iterations;

				solver_info["time_checking_for_nan_inf"] =
					m_line_search->checking_for_nan_inf_time / per_iteration;
				solver_info["time_broad_phase_ccd"] =
					m_line_search->broad_phase_ccd_time / per_iteration;
				solver_info["time_ccd"] = m_line_search->ccd_time / per_iteration;
				// Remove double counting
				solver_info["time_classical_line_search"] =
					(m_line_search->classical_line_search_time
					 - m_line_search->constraint_set_update_time)
					/ per_iteration;
				solver_info["time_line_search_constraint_set_update"] =
					m_line_search->constraint_set_update_time / per_iteration;
			}
		}

		void log_times()
		{
			polyfem::logger().debug("[timing] grad {}s, assembly {}s, inverting {}s, line_search {}s, constraint_set_update {}s, obj_fun {}s, checking_for_nan_inf {}s, broad_phase_ccd {}s, ccd {}s, classical_line_search {}s",
									grad_time,
									assembly_time,
									inverting_time,
									line_search_time,
									constraint_set_update_time + (m_line_search ? m_line_search->constraint_set_update_time : 0),
									obj_fun_time,
									m_line_search ? m_line_search->checking_for_nan_inf_time : 0,
									m_line_search ? m_line_search->broad_phase_ccd_time : 0,
									m_line_search ? m_line_search->ccd_time : 0,
									m_line_search ? m_line_search->classical_line_search_time : 0);
		}
	};
} // namespace cppoptlib
