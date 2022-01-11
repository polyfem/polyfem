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
			criteria.fDelta = solver_params.value("fDelta", -1);
			criteria.gradNorm = solver_params.value("gradNorm", 1e-8);
			criteria.iterations = solver_params.value("nl_iterations", 3000);

			use_gradient_norm = solver_params.value("useGradNorm", true);
			normalize_gradient = solver_params.value("relativeGradient", false);
			use_grad_norm_tol = solver_params.value("use_grad_norm_tol", -1);
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

			// objFunc.set_ccd_max_iterations(objFunc.max_ccd_max_iterations() / 10);

			// ---------------------------
			// Initialize the minimization
			// ---------------------------

			reset(objFunc, x); // place for children to initialize their fields

			TVector grad = TVector::Zero(x.rows());
			TVector delta_x = TVector::Zero(x.rows());

			// double factor = 1e-5;

			// Set these to nan to indicate they have not been computed yet
			double old_energy = std::nan("");
			double first_energy = std::nan("");

			double first_grad_norm = -1;

			Timer total_time("Non-linear solver time");
			total_time.start();

			m_line_search->use_grad_norm_tol = use_grad_norm_tol;

			do
			{
				// ----------------
				// Compute gradient
				// ----------------
				{
					POLYFEM_SCOPED_TIMER("[timing] constrain set update {}s", constrain_set_update_time);
					objFunc.solution_changed(x);
				}

				{
					POLYFEM_SCOPED_TIMER("[timing] compute gradient {}s", grad_time);
					objFunc.gradient(x, grad);
				}

				const double grad_norm = grad.norm();
				if (std::isnan(grad_norm))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().error("Gradient is nan; stopping");
					m_error_code = ErrorCode::NanEncountered;
					break;
				}

				if (first_grad_norm < 0)
					first_grad_norm = grad_norm;

				// ------------------------
				// Compute update direction
				// ------------------------

				// Compute a Δx to update the variable
				compute_update_direction(objFunc, x, grad, delta_x);

				const double delta_x_norm = delta_x.norm();
				if (std::isnan(delta_x_norm))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().error("Δx is nan; stopping");
					m_error_code = ErrorCode::NanEncountered;
					break;
				}

				// ---------------------
				// Check for convergence
				// ---------------------

				const double energy = objFunc.value(x);
				if (!std::isfinite(energy))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().error("f(x) is nan or inf; stopping");
					m_error_code = ErrorCode::NanEncountered;
					break;
				}
				this->m_current.fDelta = std::abs(old_energy - energy) / std::abs(old_energy);

				// If the gradient norm is really small we found a minimum
				if (!use_gradient_norm && !use_gradient_descent && grad_norm > 1e-13)
				{
					// Use the maximum absolute displacement value divided by the timestep,
					// so the units are in velocity units.
					// TODO: Set this to the actual timestep
					double dt = 1;
					// TODO: Also divide by the world scale to make this criteria scale invariant.
					this->m_current.gradNorm = delta_x.template lpNorm<Eigen::Infinity>() / dt;
				}
				else
				{
					this->m_current.gradNorm = grad_norm / (normalize_gradient ? first_grad_norm : 1);
				}

				this->m_status = checkConvergence(this->m_stop, this->m_current);

				old_energy = energy;
				if (std::isnan(first_energy))
				{
					first_energy = energy;
				}

				if (!use_gradient_norm && this->m_status != Status::Continue && this->m_status != Status::IterationLimit && grad_norm > 1e-2)
				{
					polyfem::logger().debug("[{}] converged with large gradient (||∇f||={}); continuing", name(), grad_norm);
					this->m_status = Status::Continue;
				}

				// Converge on the first iteration iff the line search failed and the gradient is small
				if ((use_gradient_descent || this->m_current.iterations > 0) && this->m_status != Status::Continue)
				{
					// converged
					break;
				}

				// ---------------
				// Variable update
				// ---------------

				// Perform a line_search to compute step scale
				double rate = line_search(x, delta_x, objFunc);
				if (std::isnan(rate))
				{
					// use_gradient_descent set by line_search upon failure
					if (use_gradient_descent)
						continue; // Try the step again with gradient descent
					else
						break; // Line search failed twice, so quit!
				}

				x += rate * delta_x;

				// -----------
				// Post update
				// -----------

				use_gradient_descent = false; // Reset this for the next iterations

				const double step = (rate * delta_x).norm();
				if (this->m_status == Status::Continue && step < 1e-10)
				{
					handle_small_step(step);
				}

				if (objFunc.stop(x))
				{
					this->m_status = Status::UserDefined;
					m_error_code = ErrorCode::Success;
					polyfem::logger().debug("Objective decided to stop");
				}

				// if(rate >= 1 && next_hessian == this->m_current.iterations)
				//	next_hessian += 2;

				objFunc.post_step(x);

				polyfem::logger().debug(
					"[{}] iter={:} f={} ‖∇f‖={} ‖Δx‖={} Δx⋅∇f(x)={} g={} tol={} rate={} ‖step‖={}",
					name(), this->m_current.iterations, energy, grad_norm, delta_x_norm, delta_x.dot(grad),
					this->m_current.gradNorm, this->m_stop.gradNorm, rate, step);
				++this->m_current.iterations;
			} while (objFunc.callback(this->m_current, x) && (this->m_status == Status::Continue));

			total_time.stop();

			// -----------
			// Log results
			// -----------

			std::string msg = "Finished";
			spdlog::level::level_enum level = spdlog::level::info;
			if (this->m_status == Status::IterationLimit)
			{
				msg = "Reached iteration limit";
				level = spdlog::level::warn;
			}
			else if (this->m_current.iterations == 0)
			{
				msg = "Unable to take a step";
				level = this->m_status == Status::UserDefined ? spdlog::level::err : spdlog::level::warn;
			}
			else if (this->m_status == Status::UserDefined)
			{
				msg = "Failed to find minimizer";
				level = spdlog::level::err;
			}
			polyfem::logger().log(
				level, "[{}] {}, took {}s (niters={} f={} ||∇f||={} ||Δx||={} Δx⋅∇f(x)={} g={} tol={})",
				name(), msg, total_time.getElapsedTimeInSec(), this->m_current.iterations, old_energy, grad.norm(), delta_x.norm(),
				delta_x.dot(grad), this->m_current.gradNorm, this->m_stop.gradNorm);

			log_times();
			update_solver_info();
		}

		double line_search(const TVector &x, const TVector &delta_x, ProblemType &objFunc)
		{
			POLYFEM_SCOPED_TIMER("[timing] line search {}s", line_search_time);

			if (!m_line_search)
				return 1e-1;

			double rate = m_line_search->line_search(x, delta_x, objFunc);

			if (std::isnan(rate) && !use_gradient_descent)
			{
				use_gradient_descent = true;
				polyfem::logger().log(
					this->m_current.iterations > 0 ? spdlog::level::err : spdlog::level::warn,
					"Line search failed; reverting to gradient descent");
				this->m_status = Status::Continue;
			}
			else if (std::isnan(rate) && use_gradient_descent)
			{
				use_gradient_descent = false;
				polyfem::logger().log(
					this->m_current.iterations > 0 ? spdlog::level::err : spdlog::level::warn,
					"Line search failed on gradient descent; stopping");
				this->m_status = Status::UserDefined;
				m_error_code = ErrorCode::NanEncountered;
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
			use_gradient_descent = false;
		}

		// Compute the search/update direction
		virtual void compute_update_direction(ProblemType &objFunc, const TVector &x_vec, const TVector &grad, TVector &direction) = 0;

		// Special handling for small steps
		virtual void handle_small_step(double step) {}

	protected:
		const json solver_params;

		std::shared_ptr<polyfem::LineSearch<ProblemType>> m_line_search;

		bool use_gradient_descent;

		ErrorCode m_error_code;
		bool use_gradient_norm;
		bool normalize_gradient;
		double use_grad_norm_tol;

		json solver_info;

		double grad_time;
		double assembly_time;
		double inverting_time;
		double line_search_time;
		double constrain_set_update_time;
		double obj_fun_time;

		void reset_times()
		{
			grad_time = 0;
			assembly_time = 0;
			inverting_time = 0;
			line_search_time = 0;
			obj_fun_time = 0;
			constrain_set_update_time = 0;
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

			grad_time /= per_iteration;
			assembly_time /= per_iteration;
			inverting_time /= per_iteration;
			line_search_time /= per_iteration;

			if (m_line_search)
			{
				constrain_set_update_time += m_line_search->constrain_set_update_time;
			}
			constrain_set_update_time /= per_iteration;
			obj_fun_time /= per_iteration;

			solver_info["time_grad"] = grad_time;
			solver_info["time_assembly"] = assembly_time;
			solver_info["time_inverting"] = inverting_time;
			solver_info["time_line_search"] = line_search_time;
			solver_info["time_constrain_set_update"] = constrain_set_update_time;
			solver_info["time_obj_fun"] = obj_fun_time;

			if (m_line_search)
			{
				solver_info["time_chekcing_for_nan_inf"] =
					m_line_search->checking_for_nan_inf_time / per_iteration;
				solver_info["time_broad_phase_ccd"] =
					m_line_search->broad_phase_ccd_time / per_iteration;
				solver_info["time_ccd"] = m_line_search->ccd_time / per_iteration;
				solver_info["time_classical_line_search"] =
					m_line_search->classical_line_search_time / per_iteration;
			}
		}

		void log_times()
		{
			polyfem::logger().debug("[timing] grad {}s, assembly {}s, inverting {}s, line_search {}s, constrain_set_update {}s, obj_fun {}s, chekcing_for_nan_inf {}s, broad_phase_ccd {}s, ccd {}s, classical_line_search {}s",
									grad_time,
									assembly_time,
									inverting_time,
									line_search_time,
									constrain_set_update_time + (m_line_search ? m_line_search->constrain_set_update_time : 0),
									obj_fun_time,
									m_line_search ? m_line_search->checking_for_nan_inf_time : 0,
									m_line_search ? m_line_search->broad_phase_ccd_time : 0,
									m_line_search ? m_line_search->ccd_time : 0,
									m_line_search ? m_line_search->classical_line_search_time : 0);
		}
	};
} // namespace cppoptlib
