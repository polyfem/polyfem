#pragma once

#include "LineSearch.hpp"
#include "ArmijoLineSearch.hpp"
#include "BacktrackingLineSearch.hpp"
#include "CppOptArmijoLineSearch.hpp"
#include "MoreThuenteLineSearch.hpp"

#include <fstream>

namespace polyfem
{
	namespace solver
	{
		namespace line_search
		{
			template <typename ProblemType>
			std::shared_ptr<LineSearch<ProblemType>> LineSearch<ProblemType>::construct_line_search(const std::string &name)
			{
				if (name == "armijo" || name == "Armijo")
				{
					return std::make_shared<ArmijoLineSearch<ProblemType>>();
				}
				else if (name == "armijo_alt" || name == "ArmijoAlt")
				{
					return std::make_shared<CppOptArmijoLineSearch<ProblemType>>();
				}
				else if (name == "bisection" || name == "Bisection")
				{
					logger().warn("{} linesearch was renamed to \"backtracking\"; using backtracking line-search", name);
					return std::make_shared<BacktrackingLineSearch<ProblemType>>();
				}
				else if (name == "backtracking" || name == "Backtracking")
				{
					return std::make_shared<BacktrackingLineSearch<ProblemType>>();
				}
				else if (name == "more_thuente" || name == "MoreThuente")
				{
					return std::make_shared<MoreThuenteLineSearch<ProblemType>>();
				}
				else if (name == "none")
				{
					return nullptr;
				}
				else
				{
					const std::string msg = fmt::format("Unknown line search {}!", name);
					logger().error(msg);
					throw std::invalid_argument(msg);
				}
			}

			template <typename ProblemType>
			void LineSearch<ProblemType>::save_sampled_values(
				const std::string &filename,
				const typename ProblemType::TVector &x,
				const typename ProblemType::TVector &delta_x,
				ProblemType &objFunc,
				const double starting_step_size,
				const int num_samples)
			{
				std::ofstream samples(filename, std::ios::out);
				if (!samples.is_open())
				{
					spdlog::error("Unable to save sampled values to file \"{}\" !", filename);
					return;
				}

				samples << "alpha,f(x + alpha * delta_x),valid,decrease\n";

				objFunc.solution_changed(x);
				double fx = objFunc.value(x);

				Eigen::VectorXd alphas = Eigen::VectorXd::LinSpaced(2 * num_samples - 1, -starting_step_size, starting_step_size);
				for (int i = 0; i < alphas.size(); i++)
				{
					typename ProblemType::TVector new_x = x + alphas[i] * delta_x;
					objFunc.solution_changed(new_x);
					double fxi = objFunc.value(new_x);
					samples << alphas[i] << ","
							<< fxi << ","
							<< (objFunc.is_step_valid(x, new_x) ? "true" : "false") << ","
							<< (fxi < fx ? "true" : "false") << "\n";
				}

				samples.close();
			}

			template <typename ProblemType>
			double LineSearch<ProblemType>::compute_nan_free_step_size(
				const TVector &x,
				const TVector &delta_x,
				ProblemType &objFunc,
				const double starting_step_size,
				const double rate)
			{
				double step_size = starting_step_size;
				TVector new_x = x + step_size * delta_x;

				// Find step that does not result in nan or infinite energy
				while (step_size > min_step_size && cur_iter < max_step_size_iter)
				{
					// Compute the new energy value without contacts
					// TODO: removed only elastic
					const double energy = objFunc.value(new_x);
					const bool is_step_valid = objFunc.is_step_valid(x, new_x);

					if (!std::isfinite(energy) || !is_step_valid)
					{
						step_size *= rate;
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

			template <typename ProblemType>
			double LineSearch<ProblemType>::compute_collision_free_step_size(
				const TVector &x,
				const TVector &delta_x,
				ProblemType &objFunc,
				const double starting_step_size)
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

				{ // clang-format off
				//#pragma STDC FENV_ACCESS ON
				const int current_round = std::fegetround();
				std::fesetround(FE_DOWNWARD);
				step_size *= max_step_size; // TODO: check me if correct
				std::fesetround(current_round);
				} // clang-format on

				// logger().trace("\t\tpre TOI={}, ss={}", max_step_size, step_size);

				// while (max_step_size != 1)
				// {
				// 	new_x = x + step_size * delta_x;
				// 	max_step_size = objFunc.max_step_size(x, new_x);
				//
				// 	std::fesetround(FE_DOWNWARD);
				// 	step_size *= max_step_size; // TODO: check me if correct
				// 	std::fesetround(current_roudn);
				// 	if (max_step_size != 1)
				// 		logger().trace("\t\trepeating TOI={}, ss={}", max_step_size, step_size);
				// }

				return step_size;
			}

			// #ifndef NDEBUG
			// 			template <typename ProblemType>
			// 			double LineSearch<ProblemType>::compute_debug_collision_free_step_size(
			// 				const TVector &x,
			// 				const TVector &delta_x,
			// 				ProblemType &objFunc,
			// 				const double starting_step_size,
			// 				const double rate)
			// 			{
			// 				double step_size = starting_step_size;

			// 				TVector new_x = x + step_size * delta_x;
			// 				{
			// 					POLYFEM_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
			// 					objFunc.solution_changed(new_x);
			// 				}

			// 				// safe guard check
			// 				while (!objFunc.is_step_collision_free(x, new_x))
			// 				{
			// 					logger().error("step is not collision free!!");
			// 					step_size *= rate;
			// 					new_x = x + step_size * delta_x;
			// 					{
			// 						POLYFEM_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
			// 						objFunc.solution_changed(new_x);
			// 					}
			// 				}
			// 				assert(objFunc.is_step_collision_free(x, new_x));

			// 				return step_size;
			// 			}
			// #endif
		} // namespace line_search
	}     // namespace solver
} // namespace polyfem
