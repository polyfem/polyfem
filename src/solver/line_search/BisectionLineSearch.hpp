#pragma once

#include <polyfem/LineSearch.hpp>

#include <polyfem/Timer.hpp>

#include <cfenv>

namespace polyfem
{
	template <typename ProblemType>
	class BisectionLineSearch : public LineSearch<ProblemType>
	{
	public:
		using Superclass = LineSearch<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		double line_search(
			const TVector &x,
			const TVector &delta_x,
			ProblemType &objFunc) override
		{
			// ----------------
			// Begin linesearch
			// ----------------

			double old_energy, step_size;
			{
				POLYFEM_SCOPED_TIMER_NO_GLOBAL("[timing] LS begin {}s");

				cur_iter = 0;

				old_energy = objFunc.value(x);
				if (std::isnan(old_energy))
				{
					logger().error("Original energy in line search is nan!");
					return std::nan("");
				}

				step_size = objFunc.heuristic_max_step(delta_x);
			}

			// ----------------------------
			// Find finite energy step size
			// ----------------------------

			{
				POLYFEM_SCOPED_TIMER("[timing] LS compute finite energy step size {}s", this->checking_for_nan_inf_time);
				step_size = compute_nan_free_step_size(x, delta_x, objFunc, step_size);
				if (std::isnan(step_size))
					return std::nan("");
			}

			const double nan_free_step_size = step_size;

			// -----------------------------
			// Find collision-free step size
			// -----------------------------

			{
				POLYFEM_SCOPED_TIMER("[timing] CCD broad-phase {}s", this->broad_phase_ccd_time);
				TVector new_x = x + step_size * delta_x;
				objFunc.line_search_begin(x, new_x);
			}

			{
				POLYFEM_SCOPED_TIMER("[timing] CCD narrow-phase {}s", this->ccd_time);
				step_size = compute_collision_free_step_size(x, delta_x, objFunc, step_size);
				if (std::isnan(step_size))
					return std::nan("");
			}

			const double collision_free_step_size = step_size;

			// ----------------------
			// Find descent step size
			// ----------------------

			{
				POLYFEM_SCOPED_TIMER("[timing] energy min in LS {}s", this->classical_line_search_time);
				step_size = compute_descent_step_size(x, delta_x, objFunc, old_energy, step_size);
				if (std::isnan(step_size))
				{
					// Superclass::save_sampled_values("failed-line-search-values.csv", x, delta_x, objFunc);
					return std::nan("");
				}
			}

			const double descent_step_size = step_size;

#ifndef NDEBUG
			// -------------
			// CCD safeguard
			// -------------

			{
				POLYFEM_SCOPED_TIMER_NO_GLOBAL("[timing] safeguard in LS {}s");
				step_size = compute_debug_collision_free_step_size(x, delta_x, objFunc, step_size);
			}

			const double debug_collision_free_step_size = step_size;
#endif

			{
				POLYFEM_SCOPED_TIMER_NO_GLOBAL("[timing] LS end {}s");
				objFunc.line_search_end();
			}

			logger().debug(
				"Line search finished (nan_free_step_size={} collision_free_step_size={} descent_step_size={} final_step_size={})",
				nan_free_step_size, collision_free_step_size, descent_step_size, step_size);

			return step_size;
		}

	protected:
		double min_step_size = 0;
		int max_step_size_iter = 100; // std::numeric_limits<int>::max();
		int cur_iter = 0;

		double compute_nan_free_step_size(
			const TVector &x,
			const TVector &delta_x,
			ProblemType &objFunc,
			const double starting_step_size = 1)
		{
			double step_size = starting_step_size;
			TVector new_x = x + step_size * delta_x;

			// Find step that does not result in nan or infinite energy
			while (step_size > min_step_size && cur_iter < max_step_size_iter)
			{
				// Compute the new energy value without contacts
				const double energy = objFunc.value(new_x, /*only_elastic=*/true);
				const bool is_step_valid = objFunc.is_step_valid(x, new_x);

				if (!std::isfinite(energy) || !is_step_valid)
				{
					step_size /= 2.0;
					new_x = x + step_size * delta_x;
				}
				else
				{
					break;
				}
				cur_iter++;
			}

			if (cur_iter >= max_step_size_iter || step_size <= min_step_size)
			{
				logger().error(
					"Line search failed to find a valid finite energy step (cur_iter={:d} step_size={:g})!",
					cur_iter, step_size);
				return std::nan("");
			}

			return step_size;
		}

		double compute_collision_free_step_size(
			const TVector &x,
			const TVector &delta_x,
			ProblemType &objFunc,
			const double starting_step_size = 1)
		{
			double step_size = starting_step_size;
			TVector new_x = x + step_size * delta_x;

			// Find step that is collision free
			double max_step_size = objFunc.max_step_size(x, new_x);
			if (max_step_size == 0)
			{
				logger().error("Line search failed because CCD produced a stepsize of zero!");
				objFunc.line_search_end();
				return std::nan("");
			}

#pragma STDC FENV_ACCESS ON
			const int current_round = std::fegetround();
			std::fesetround(FE_DOWNWARD);
			step_size *= max_step_size; // TODO: check me if correct
			std::fesetround(current_round);
			// logger().trace("\t\tpre TOI={}, ss={}", tmp, step_size);

			// while (tmp != 1)
			// {
			// 	new_x = x + step_size * delta_x;
			// 	tmp = objFunc.max_step_size(x, new_x);
			//
			// 	std::fesetround(FE_DOWNWARD);
			// 	step_size *= tmp; // TODO: check me if correct
			// 	std::fesetround(current_roudn);
			// 	if (tmp != 1)
			// 		logger().trace("\t\trepeating TOI={}, ss={}", tmp, step_size);
			// }

			return step_size;
		}

		double compute_descent_step_size(
			const TVector &x,
			const TVector &delta_x,
			ProblemType &objFunc,
			const double old_energy,
			const double starting_step_size = 1)
		{
			double step_size = starting_step_size;

			// Find step that reduces the energy
			double cur_energy = std::nan("");
			bool is_step_valid = false;
			while (step_size > min_step_size && cur_iter < max_step_size_iter)
			{
				TVector new_x = x + step_size * delta_x;

				{
					POLYFEM_SCOPED_TIMER("[timing] constrain set update in LS {}s", this->constrain_set_update_time);
					objFunc.solution_changed(new_x);
				}

				cur_energy = objFunc.value(new_x);
				is_step_valid = objFunc.is_step_valid(x, new_x);

				logger().trace("ls it: {} delta: {} invalid: {} ", cur_iter, (cur_energy - old_energy), !is_step_valid);

				// if (!std::isfinite(cur_energy) || (cur_energy >= old_energy && fabs(cur_energy - old_energy) > 1e-12) || !is_step_valid)
				if (!std::isfinite(cur_energy) || cur_energy >= old_energy || !is_step_valid)
				{
					step_size /= 2.0;
					// max_step_size should return a collision free step
					// assert(objFunc.is_step_collision_free(x, new_x));
				}
				else
				{
					break;
				}
				cur_iter++;
			}

			if (cur_iter >= max_step_size_iter || step_size <= min_step_size)
			{
				logger().warn(
					"Line search failed to find descent step (f(x)={:g} f(x+αΔx)={:g} α_CCD={:g} α={:g}, ||Δx||={:g} is_step_valid={} iter={:d})",
					old_energy, cur_energy, starting_step_size, step_size, delta_x.norm(),
					is_step_valid ? "true" : "false", cur_iter);
				objFunc.line_search_end();
				return std::nan("");
			}

			return step_size;
		}

#ifndef NDEBUG
		double compute_debug_collision_free_step_size(
			const TVector &x,
			const TVector &delta_x,
			ProblemType &objFunc,
			const double starting_step_size = 1)
		{
			double step_size = starting_step_size;

			TVector new_x = x + step_size * delta_x;
			{
				POLYFEM_SCOPED_TIMER("[timing] constrain set update in LS {}s", this->constrain_set_update_time);
				objFunc.solution_changed(new_x);
			}

			// safe guard check
			while (!objFunc.is_step_collision_free(x, new_x))
			{
				logger().error("step is not collision free!!");
				step_size /= 2;
				new_x = x + step_size * delta_x;
				{
					POLYFEM_SCOPED_TIMER("[timing] constrain set update in LS {}s", this->constrain_set_update_time);
					objFunc.solution_changed(new_x);
				}
			}
			assert(objFunc.is_step_collision_free(x, new_x));

			return step_size;
		}
#endif
	};
} // namespace polyfem
