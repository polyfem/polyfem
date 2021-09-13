#pragma once

#include <polyfem/LineSearch.hpp>

#include <igl/Timer.h>

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

		double linesearch(
			const TVector &x,
			const TVector &grad,
			ProblemType &objFunc) override
		{
			static const int MAX_STEP_SIZE_ITER = std::numeric_limits<int>::max();
			static const double MIN_STEP_SIZE = 0;

			igl::Timer timer, solution_changed_timer;

			timer.start();

			const double old_energy = objFunc.value(x);
			if (std::isnan(old_energy))
			{
				logger().error("Original energy in line search is nan!");
				return std::nan("");
			}
			int cur_iter = 0;

			double step_size = objFunc.heuristic_max_step(grad);
			TVector new_x = x + grad * step_size;

			timer.stop();
			logger().trace("\t\tLS begin {}s", timer.getElapsedTimeInSec());

			timer.start();
			// Find step that does not result in nan or infinite energy
			while (step_size > MIN_STEP_SIZE && cur_iter < MAX_STEP_SIZE_ITER)
			{
				// Compute the new energy value without contacts
				double cur_e = objFunc.value(new_x, /*only_elastic=*/true);
				const bool valid = objFunc.is_step_valid(x, new_x);

				if (!std::isfinite(cur_e) || !valid)
				{
					step_size /= 2.;
					new_x = x + step_size * grad;
				}
				else
				{
					break;
				}
				cur_iter++;
			}
			timer.stop();
			logger().trace("\t\tfist loop in LS {}s, checking for nan or inf. Stepsize = {:g}", timer.getElapsedTimeInSec(), step_size);
			this->checking_for_nan_inf_time += timer.getElapsedTimeInSec();

			if (cur_iter >= MAX_STEP_SIZE_ITER || step_size <= MIN_STEP_SIZE)
			{
				logger().error(
					"Line search failed to find a valid finite energy step (old_energy={:g} cur_iter={:d} step_size={:g})!",
					old_energy, cur_iter, step_size);
				return std::nan("");
			}

			timer.start();
			objFunc.line_search_begin(x, new_x);
			timer.stop();
			logger().trace("\t\tbroad phase CCD {}s", timer.getElapsedTimeInSec());
			this->broad_phase_ccd_time += timer.getElapsedTimeInSec();

			timer.start();
			// Find step that is collision free
			double tmp = objFunc.max_step_size(x, new_x);
			if (tmp == 0)
			{
				logger().error("Line search failed because CCD produced a stepsize of zero!");
				objFunc.line_search_end();
				return std::nan("");
			}

#pragma STDC FENV_ACCESS ON
			const int current_roudn = std::fegetround();
			std::fesetround(FE_DOWNWARD);
			step_size *= tmp; // TODO: check me if correct
			std::fesetround(current_roudn);
			// logger().trace("\t\tpre TOI={}, ss={}", tmp, step_size);

			// while (tmp != 1)
			// {
			// 	new_x = x + step_size * grad;
			// 	tmp = objFunc.max_step_size(x, new_x);

			// 	std::fesetround(FE_DOWNWARD);
			// 	step_size *= tmp; // TODO: check me if correct
			// 	std::fesetround(current_roudn);
			// 	if (tmp != 1)
			// 		logger().trace("\t\trepeating TOI={}, ss={}", tmp, step_size);
			// }

			new_x = x + step_size * grad;

			timer.stop();
			logger().trace("\t\tCCD in LS {}s, step={}", timer.getElapsedTimeInSec(), step_size);
			this->ccd_time += timer.getElapsedTimeInSec();

			timer.start();
			objFunc.solution_changed(new_x);
			timer.stop();
			logger().trace("\t\tconstrain set update in LS {}s", timer.getElapsedTimeInSec());
			this->constrain_set_update_time += timer.getElapsedTimeInSec();

			// Find step that reduces the energy
			timer.start();
			double cur_energy = std::nan("");
			bool is_step_valid = false;
			while (step_size > MIN_STEP_SIZE && cur_iter < MAX_STEP_SIZE_ITER)
			{
				cur_energy = objFunc.value(new_x);
				is_step_valid = objFunc.is_step_valid(x, new_x);

				logger().trace("ls it: {} delta: {} invalid: {} ", cur_iter, (cur_energy - old_energy), !is_step_valid);
				// if (!std::isfinite(cur_energy) || (cur_energy >= old_energy && fabs(cur_energy - old_energy) > 1e-12) || !is_step_valid)
				if (!std::isfinite(cur_energy) || cur_energy >= old_energy || !is_step_valid)
				{
					step_size /= 2.;
					new_x = x + step_size * grad;
					//max_step_size should return a collision free step
					// assert(objFunc.is_step_collision_free(x, new_x));

					solution_changed_timer.start();
					objFunc.solution_changed(new_x);
					solution_changed_timer.stop();
					logger().trace("\t\t\tconstrain set update in LS {}s", solution_changed_timer.getElapsedTimeInSec());
				}
				else
				{
					break;
				}
				cur_iter++;
			}
			timer.stop();
			logger().trace("\t\tenergy min in LS {}s", timer.getElapsedTimeInSec());
			this->classical_linesearch_time += timer.getElapsedTimeInSec();

			if (cur_iter >= MAX_STEP_SIZE_ITER || step_size <= MIN_STEP_SIZE)
			{
				logger().error(
					"Line search failed to find descent step "
					"(old_energy={:.16g} cur_energy={:.16g} is_step_valid={} cur_iter={:d} step_size={:g})",
					old_energy, cur_energy, is_step_valid ? "true" : "false", cur_iter, step_size);
				objFunc.line_search_end();
				return std::nan("");
			}

			logger().trace("final step ratio {}, step {}", tmp, step_size);

#ifndef NDEBUG
			// safe guard check
			timer.start();
			while (!objFunc.is_step_collision_free(x, new_x))
			{
				logger().error("step is not collision free!!");
				step_size /= 2;
				new_x = x + step_size * grad;
				// timer.start();
				objFunc.solution_changed(new_x);
				// timer.stop();
				// logger().trace("\tconstrain set update in LS {}s", timer.getElapsedTimeInSec());
			}
			assert(objFunc.is_step_collision_free(x, new_x));
			timer.stop();
			logger().trace("\t\tsafeguard in LS {}s", timer.getElapsedTimeInSec());
#endif

			timer.start();
			objFunc.line_search_end();
			timer.stop();
			logger().trace("\t\tLS end {}s", timer.getElapsedTimeInSec());

			return step_size;
		}
	};
} // namespace polyfem
