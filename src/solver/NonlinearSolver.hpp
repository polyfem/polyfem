#pragma once

#include <polyfem/Common.hpp>

// Line search methods
#include <polyfem/LineSearch.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

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

		NonlinearSolver(const json &solver_params)
			: solver_params(solver_params)
		{
			auto criteria = this->criteria();
			criteria.fDelta = solver_params.count("fDelta") ? double(solver_params["fDelta"]) : 1e-9;
			criteria.gradNorm = solver_params.count("gradNorm") ? double(solver_params["gradNorm"]) : 1e-8;
			criteria.iterations = solver_params.count("nl_iterations") ? int(solver_params["nl_iterations"]) : 3000;

			use_gradient_norm = solver_params.count("useGradNorm") ? bool(solver_params["useGradNorm"]) : true;
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

			igl::Timer timer;

			// objFunc.set_ccd_max_iterations(objFunc.max_ccd_max_iterations() / 10);

			// ---------------------------
			// Initialize the minimization
			// ---------------------------

			timer.start();
			reset(objFunc, x); // place for children to initialize their fields

			TVector grad = TVector::Zero(x.rows());
			TVector delta_x = TVector::Zero(x.rows());

			// double factor = 1e-5;

			// Set these to nan to indicate they have not been computed yet
			double old_energy = std::nan("");
			double first_energy = std::nan("");
			timer.stop();
			polyfem::logger().trace("\tinitialization took {}s", timer.getElapsedTimeInSec());

			timer.start();
			objFunc.solution_changed(x);
			timer.stop();
			polyfem::logger().debug("\tconstrain set update {}s", timer.getElapsedTimeInSec());
			constrain_set_update_time += timer.getElapsedTimeInSec();

			timer.start();
			objFunc.gradient(x, grad);
			this->m_current.gradNorm = grad.norm();
			timer.stop();
			polyfem::logger().debug("\tgrad time {}s norm: {}", timer.getElapsedTimeInSec(), this->m_current.gradNorm);
			grad_time += timer.getElapsedTimeInSec();

			if (std::isnan(this->m_current.gradNorm))
			{
				this->m_status = Status::UserDefined;
				polyfem::logger().debug("stopping because first grad is nan");
				m_error_code = -10;
				return;
			}

			do
			{
				// ---------------
				// Variable update
				// ---------------

				// Computea delta_x to update the variable
				compute_search_direction(objFunc, x, grad, delta_x);

				// Perform a linesearch to compute step scale
				double rate = linesearch(x, delta_x, objFunc);
				if (line_search_failed)
					continue; // Try the solve again with gradient descent
				else if (std::isnan(rate))
					break; // Line search failed twice, so quit!

				x += rate * delta_x;

				// ------------------
				// Recompute gradient
				// ------------------

				timer.start();
				objFunc.solution_changed(x);
				timer.stop();
				polyfem::logger().debug("\tconstrain set update {}s", timer.getElapsedTimeInSec());
				obj_fun_time += timer.getElapsedTimeInSec();

				timer.start();
				objFunc.gradient(x, grad);
				timer.stop();
				polyfem::logger().debug("\tgrad time {}s norm: {}", timer.getElapsedTimeInSec(), grad.norm());
				grad_time += timer.getElapsedTimeInSec();

				// ---------------------
				// Check for convergence
				// ---------------------

				const double energy = objFunc.value(x);
				const double step = (rate * delta_x).norm();

				++this->m_current.iterations;
				this->m_current.fDelta = 1; //std::abs(old_energy - energy) / std::abs(old_energy);
				this->m_current.gradNorm = grad.norm();
				// If the gradient norm is really small we found a minimum
				if (this->m_current.gradNorm > 1e-13 && !use_gradient_norm)
				{
					// Use the maximum absolute displacement value divided by the timestep,
					// so the units are in velocity units.
					// TODO: Set this to the actual timestep
					double dt = 1;
					// TODO: Also divide by the world scale to make this criteria scale invariant.
					this->m_current.gradNorm = delta_x.template lpNorm<Eigen::Infinity>() / dt;
				}
				this->m_status = checkConvergence(this->m_stop, this->m_current);
				old_energy = energy;
				if (std::isnan(first_energy))
				{
					first_energy = energy;
				}

				if (!std::isfinite(energy))
				{
					this->m_status = Status::UserDefined;
					polyfem::logger().debug("stopping because obj func is nan or inf");
					m_error_code = -10;
				}

				if (this->m_status == Status::Continue && step < 1e-10)
				{
					handle_small_step(step);
				}

				if (objFunc.stop(x))
				{
					this->m_status = Status::UserDefined;
					m_error_code = 0;
					polyfem::logger().debug("\tObjective decided to stop");
				}

				//if(rate >= 1 && next_hessian == this->m_current.iterations)
				//	next_hessian += 2;

				objFunc.post_step(x);

				polyfem::logger().debug("\titer: {}, f = {}, ||g||_2 = {}, rate = {}, ||step|| = {}, dot = {}",
										this->m_current.iterations, energy, this->m_current.gradNorm, rate, step, delta_x.dot(grad) / grad.norm());
				delta_x *= -1;
			} while (objFunc.callback(this->m_current, x) && (this->m_status == Status::Continue));

			if (this->m_status == Status::IterationLimit)
			{
				polyfem::logger().warn(
					"{} reached iteration limit (niters = {}, f = {}, ||g||_2 = {})",
					name(), this->m_current.iterations, old_energy, this->m_current.gradNorm);
			}
			else
			{
				polyfem::logger().info(
					"{} finished (niters = {}, f = {}, ||g||_2 = {})",
					name(), this->m_current.iterations, old_energy, this->m_current.gradNorm);
			}

			log_times();
			update_solver_info();
		}

		double linesearch(const TVector &x, const TVector &delta_x, ProblemType &objFunc)
		{
			igl::Timer timer;
			timer.start();

			double rate = m_line_search ? m_line_search->linesearch(x, delta_x, objFunc) : 1e-1;

			timer.stop();
			polyfem::logger().debug("\tlinesearch time {}s", timer.getElapsedTimeInSec());
			linesearch_time += timer.getElapsedTimeInSec();

			if (std::isnan(rate) && !line_search_failed)
			{
				line_search_failed = true;
				polyfem::logger().debug("\tline search failed, reverting to gradient descent");
				this->m_status = Status::Continue;
				return rate;
			}

			line_search_failed = false;

			if (std::isnan(rate))
			{
				polyfem::logger().error("Line search failed, stopping");
				this->m_status = Status::UserDefined;
				m_error_code = -10;
				return rate;
			}

			return rate;
		}

		void getInfo(json &params)
		{
			params = solver_info;
		}

		int error_code() const { return m_error_code; }

	protected:
		// Reset the solver at the start of a minimization
		virtual void reset(ProblemType &objFunc, TVector &x)
		{
			this->m_current.reset();
			reset_times();
			m_error_code = 0;
			line_search_failed = false;
		}

		// Compute the search/update direction
		virtual void compute_search_direction(ProblemType &objFunc, const TVector &x_vec, const TVector &grad, TVector &direction) = 0;

		// Special handling for small steps
		virtual void handle_small_step(double step) {}

	protected:
		const json solver_params;

		std::shared_ptr<polyfem::LineSearch<ProblemType>> m_line_search;
		bool line_search_failed;

		int m_error_code;
		bool use_gradient_norm;

		json solver_info;

		double grad_time;
		double assembly_time;
		double inverting_time;
		double linesearch_time;
		double constrain_set_update_time;
		double obj_fun_time;

		void reset_times()
		{
			grad_time = 0;
			assembly_time = 0;
			inverting_time = 0;
			linesearch_time = 0;
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

			grad_time /= crit.iterations;
			assembly_time /= crit.iterations;
			inverting_time /= crit.iterations;
			linesearch_time /= crit.iterations;

			if (m_line_search)
			{
				constrain_set_update_time += m_line_search->constrain_set_update_time;
			}
			constrain_set_update_time /= crit.iterations;
			obj_fun_time /= crit.iterations;

			solver_info["time_grad"] = grad_time;
			solver_info["time_assembly"] = assembly_time;
			solver_info["time_inverting"] = inverting_time;
			solver_info["time_linesearch"] = linesearch_time;
			solver_info["time_constrain_set_update"] = constrain_set_update_time;
			solver_info["time_obj_fun"] = obj_fun_time;

			if (m_line_search)
			{
				solver_info["time_chekcing_for_nan_inf"] =
					m_line_search->checking_for_nan_inf_time / crit.iterations;
				solver_info["time_broad_phase_ccd"] =
					m_line_search->broad_phase_ccd_time / crit.iterations;
				solver_info["time_ccd"] = m_line_search->ccd_time / crit.iterations;
				solver_info["time_classical_linesearch"] =
					m_line_search->classical_linesearch_time / crit.iterations;
			}
		}

		void log_times()
		{
			polyfem::logger().trace("grad {}s, assembly {}s, inverting {}s, linesearch {}s, constrain_set_update {}s, obj_fun {}s, chekcing_for_nan_inf {}s, broad_phase_ccd {}s, ccd {}s, classical_linesearch {}s",
									grad_time,
									assembly_time,
									inverting_time,
									linesearch_time,
									constrain_set_update_time + (m_line_search ? m_line_search->constrain_set_update_time : 0),
									obj_fun_time,
									m_line_search ? m_line_search->checking_for_nan_inf_time : 0,
									m_line_search ? m_line_search->broad_phase_ccd_time : 0,
									m_line_search ? m_line_search->ccd_time : 0,
									m_line_search ? m_line_search->classical_linesearch_time : 0);
		}
	};
} // namespace cppoptlib
