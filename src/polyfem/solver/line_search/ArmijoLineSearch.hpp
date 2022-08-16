#pragma once

#include "LineSearch.hpp"

#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	namespace solver
	{
		namespace line_search
		{
			template <typename ProblemType>
			class ArmijoLineSearch : public LineSearch<ProblemType>
			{
			public:
				using Superclass = LineSearch<ProblemType>;
				using typename Superclass::Scalar;
				using typename Superclass::TVector;

				ArmijoLineSearch()
				{
					this->min_step_size = 1e-7;
					this->max_step_size_iter = 20; // std::numeric_limits<int>::max();
				}

				double line_search(
					const TVector &x,
					const TVector &searchDir,
					ProblemType &objFunc) override
				{
					TVector grad(x.rows());
					double f_in, alpha;
					double alpha_init = default_alpha_init;

					{
						POLYFEM_SCOPED_TIMER("LS begin");

						this->cur_iter = 0;

						f_in = objFunc.value(x);
						if (std::isnan(f_in))
						{
							logger().error("Original energy in line search is nan!");
							return std::nan("");
						}

						objFunc.gradient(x, grad);

						alpha_init = std::min(objFunc.heuristic_max_step(searchDir), alpha_init);
					}

					{
						POLYFEM_SCOPED_TIMER("LS compute finite energy step size", this->checking_for_nan_inf_time);
						alpha_init = this->compute_nan_free_step_size(x, searchDir, objFunc, alpha_init, tau);
						if (std::isnan(alpha_init))
							return std::nan("");
					}

					const double nan_free_step_size = alpha_init;

					{
						POLYFEM_SCOPED_TIMER("CCD broad-phase", this->broad_phase_ccd_time);
						const TVector x1 = x + alpha_init * searchDir;
						objFunc.line_search_begin(x, x1);
					}

					{
						POLYFEM_SCOPED_TIMER("CCD narrow-phase", this->ccd_time);
						alpha = this->compute_collision_free_step_size(x, searchDir, objFunc, alpha_init);
						if (std::isnan(alpha))
							return std::nan("");
					}

					const double collision_free_step_size = alpha;

					double f;
					bool valid;
					{
						POLYFEM_SCOPED_TIMER("energy min in LS", this->classical_line_search_time);

						TVector x1 = x + alpha * searchDir;
						{
							POLYFEM_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
							objFunc.solution_changed(x1);
						}

						objFunc.gradient(x, grad);
						const bool use_grad_norm = grad.norm() < this->use_grad_norm_tol;
						if (use_grad_norm)
							f_in = grad.squaredNorm();

						f = use_grad_norm ? grad.squaredNorm() : objFunc.value(x1);
						const double Cache = c * grad.dot(searchDir);
						valid = objFunc.is_step_valid(x, x1);

						//max_step_size should return a collision free step
						assert(objFunc.is_step_collision_free(x, x1));

						while ((std::isinf(f) || std::isnan(f) || f > f_in + alpha * Cache || !valid) && alpha > this->min_step_size && this->cur_iter <= this->max_step_size_iter)
						{
							alpha *= tau;
							x1 = x + alpha * searchDir;

							{
								POLYFEM_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
								objFunc.solution_changed(x1);
							}

							if (use_grad_norm)
							{
								objFunc.gradient(x1, grad);
								f = grad.squaredNorm();
							}
							else
								f = objFunc.value(x1);

							valid = objFunc.is_step_valid(x, x1);

							//max_step_size should return a collision free step
							assert(objFunc.is_step_collision_free(x, x1));

							logger().trace("ls it: {} f: {} (f_in + alpha * Cache): {} invalid: {} ", this->cur_iter, f, f_in + alpha * Cache, !valid);

							this->cur_iter++;
						}
					}

					const double descent_step_size = alpha;

					if (this->cur_iter >= this->max_step_size_iter || alpha <= this->min_step_size)
					{
						{
							POLYFEM_SCOPED_TIMER("LS end");
							objFunc.line_search_end();
						}

						logger().warn(
							"Line search failed to find descent step (f(x)={:g} f(x+αΔx)={:g} α_CCD={:g} α={:g}, ||Δx||={:g} is_step_valid={} iter={:d})",
							f_in, f, default_alpha_init, alpha, searchDir.norm(),
							valid ? "true" : "false", this->cur_iter);
						return std::nan("");
					}

#ifndef NDEBUG
					// -------------
					// CCD safeguard
					// -------------
					{
						POLYFEM_SCOPED_TIMER("safeguard in LS");
						alpha = this->compute_debug_collision_free_step_size(x, searchDir, objFunc, alpha, tau);
					}

					const double debug_collision_free_step_size = alpha;
#endif

					{
						POLYFEM_SCOPED_TIMER("LS end");
						objFunc.line_search_end();
					}

					logger().debug(
						"Line search finished (nan_free_step_size={} collision_free_step_size={} descent_step_size={} final_step_size={})",
						nan_free_step_size, collision_free_step_size, descent_step_size, alpha);

					return alpha;
				}

			protected:
				const double default_alpha_init = 1.0;
				const double c = 0.5;
				const double tau = 0.5;
			};
		} // namespace line_search
	}     // namespace solver
} // namespace polyfem
