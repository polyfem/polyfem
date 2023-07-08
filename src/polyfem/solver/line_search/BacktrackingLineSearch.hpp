#pragma once

#include "LineSearch.hpp"

#include <polyfem/utils/Timer.hpp>

#include <cfenv>

namespace polyfem
{
	namespace solver
	{
		namespace line_search
		{
			template <typename ProblemType>
			class BacktrackingLineSearch : public LineSearch<ProblemType>
			{
			public:
				using Superclass = LineSearch<ProblemType>;
				using typename Superclass::Scalar;
				using typename Superclass::TVector;

				BacktrackingLineSearch()
				{
					this->min_step_size = 0;
					this->max_step_size_iter = 100; // std::numeric_limits<int>::max();
				}

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
						POLYFEM_SCOPED_TIMER("LS begin");

						this->cur_iter = 0;

						old_energy = objFunc.value(x);
						if (std::isnan(old_energy))
						{
							logger().error("Original energy in line search is nan!");
							return std::nan("");
						}

						step_size = 1;
						// TODO: removed feature
						// objFunc.heuristic_max_step(delta_x);
					}

					// ----------------------------
					// Find finite energy step size
					// ----------------------------

					{
						POLYFEM_SCOPED_TIMER("LS compute finite energy step size", this->checking_for_nan_inf_time);
						step_size = this->compute_nan_free_step_size(x, delta_x, objFunc, step_size, 0.5);
						if (std::isnan(step_size))
							return std::nan("");
					}

					const double nan_free_step_size = step_size;

					// -----------------------------
					// Find collision-free step size
					// -----------------------------

					{
						POLYFEM_SCOPED_TIMER("Line Search Begin - CCD broad-phase", this->broad_phase_ccd_time);
						TVector new_x = x + step_size * delta_x;
						objFunc.line_search_begin(x, new_x);
					}

					{
						POLYFEM_SCOPED_TIMER("CCD narrow-phase", this->ccd_time);
						logger().trace("Performing narrow-phase CCD");
						step_size = this->compute_collision_free_step_size(x, delta_x, objFunc, step_size);
						if (std::isnan(step_size))
							return std::nan("");
					}

					const double collision_free_step_size = step_size;

					// ----------------------
					// Find descent step size
					// ----------------------

					{
						POLYFEM_SCOPED_TIMER("energy min in LS", this->classical_line_search_time);
						step_size = compute_descent_step_size(x, delta_x, objFunc, old_energy, step_size);
						if (std::isnan(step_size))
						{
							// Superclass::save_sampled_values("failed-line-search-values.csv", x, delta_x, objFunc);
							return std::nan("");
						}
					}

					const double descent_step_size = step_size;

					// #ifndef NDEBUG
					// 					// -------------
					// 					// CCD safeguard
					// 					// -------------

					// 					{
					// 						POLYFEM_SCOPED_TIMER("safeguard in LS");
					// 						step_size = this->compute_debug_collision_free_step_size(x, delta_x, objFunc, step_size, 0.5);
					// 					}

					// 					const double debug_collision_free_step_size = step_size;
					// #endif

					{
						POLYFEM_SCOPED_TIMER("LS end");
						objFunc.line_search_end();
					}

					logger().debug(
						"Line search finished (nan_free_step_size={} collision_free_step_size={} descent_step_size={} final_step_size={})",
						nan_free_step_size, collision_free_step_size, descent_step_size, step_size);

					return step_size;
				}

			protected:
				double compute_descent_step_size(
					const TVector &x,
					const TVector &delta_x,
					ProblemType &objFunc,
					const double old_energy_in,
					const double starting_step_size = 1)
				{
					double step_size = starting_step_size;

					TVector grad(x.rows());
					objFunc.gradient(x, grad);
					const bool use_grad_norm = grad.norm() < this->use_grad_norm_tol;

					const double old_energy = use_grad_norm ? grad.squaredNorm() : old_energy_in;

					// Find step that reduces the energy
					double cur_energy = std::nan("");
					bool is_step_valid = false;
					while (step_size > this->min_step_size && this->cur_iter < this->max_step_size_iter)
					{
						this->iterations++;

						TVector new_x = x + step_size * delta_x;

						try {
							POLYFEM_SCOPED_TIMER("solution changed - constraint set update in LS", this->constraint_set_update_time);
							objFunc.solution_changed(new_x);
						}
						catch (const std::runtime_error &e) {
							logger().warn("Failed to take step due to \"{}\", reduce step size...", e.what());

							step_size /= 2.0;
							this->cur_iter++;
							continue;
						}

						if (use_grad_norm)
						{
							objFunc.gradient(new_x, grad);
							cur_energy = grad.squaredNorm();
						}
						else
							cur_energy = objFunc.value(new_x);

						is_step_valid = objFunc.is_step_valid(x, new_x);

						logger().trace("ls it: {} delta: {} invalid: {} ", this->cur_iter, (cur_energy - old_energy), !is_step_valid);

						// if (!std::isfinite(cur_energy) || (cur_energy >= old_energy && fabs(cur_energy - old_energy) > 1e-12) || !is_step_valid)
						if (!std::isfinite(cur_energy) || cur_energy > old_energy || !is_step_valid)
						{
							step_size /= 2.0;
							// max_step_size should return a collision free step
							// assert(objFunc.is_step_collision_free(x, new_x));
						}
						else
						{
							break;
						}
						this->cur_iter++;
					}

					if (this->cur_iter >= this->max_step_size_iter || step_size <= this->min_step_size)
					{
						logger().warn(
							"Line search failed to find descent step (f(x)={:g} f(x+αΔx)={:g} α_CCD={:g} α={:g}, ||Δx||={:g} is_step_valid={} use_grad_norm={} iter={:d})",
							old_energy, cur_energy, starting_step_size, step_size, delta_x.norm(),
							is_step_valid, use_grad_norm, this->cur_iter);
						objFunc.solution_changed(x);
#ifndef NDEBUG
						// tolerance for rounding error due to multithreading
						assert(abs(old_energy_in - objFunc.value(x)) < 1e-15);
#endif
						objFunc.line_search_end();
						return std::nan("");
					}

					return step_size;
				}
			};
		} // namespace line_search
	}     // namespace solver
} // namespace polyfem
