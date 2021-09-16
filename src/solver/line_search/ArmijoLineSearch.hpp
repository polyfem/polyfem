#pragma once

#include <polyfem/LineSearch.hpp>

namespace polyfem
{
	template <typename ProblemType>
	class ArmijoLineSearch : public LineSearch<ProblemType>
	{
	public:
		using Superclass = LineSearch<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		double line_search(
			const TVector &x,
			const TVector &searchDir,
			ProblemType &objFunc) override
		{
			static const int MAX_STEP_SIZE_ITER = 12;

			double alpha_init = 1.0;
			const double c = 0.5;
			const double tau = 0.5;
			const double f_in = objFunc.value(x);

			TVector grad(x.rows());
			objFunc.gradient(x, grad);

			alpha_init = std::min(objFunc.heuristic_max_step(searchDir), alpha_init);

			TVector x1 = x + alpha_init * searchDir;

			objFunc.line_search_begin(x, x1);
			// time.start();
			objFunc.solution_changed(x1);
			// time.stop();
			// polyfem::logger().trace("\tconstrain set update in LS {}s", time.getElapsedTimeInSec());
			double alpha = std::min(alpha_init, objFunc.max_step_size(x, x1));
			// polyfem::logger().trace("inital step {}", step_size);
			if (alpha != alpha_init)
			{
				x1 = x + alpha * searchDir;
				// time.start();
				objFunc.solution_changed(x1);
				// time.stop();
				// polyfem::logger().trace("\tconstrain set update in LS {}s", time.getElapsedTimeInSec());
			}

			double f = objFunc.value(x1);

			const double Cache = c * grad.dot(searchDir);

			int cur_iter = 0;
			bool valid = objFunc.is_step_valid(x, x1);

			//max_step_size should return a collision free step
			assert(objFunc.is_step_collision_free(x, x1));

			while ((std::isinf(f) || std::isnan(f) || f > f_in + alpha * Cache || !valid) && alpha > 1e-7 && cur_iter <= MAX_STEP_SIZE_ITER)
			{
				alpha *= tau;
				x1 = x + alpha * searchDir;

				// time.start();
				objFunc.solution_changed(x1);
				// time.stop();
				// polyfem::logger().trace("\tconstrain set update in LS {}s", time.getElapsedTimeInSec());

				f = objFunc.value(x1);

				cur_iter++;
				valid = objFunc.is_step_valid(x, x1);

				//max_step_size should return a collision free step
				assert(objFunc.is_step_collision_free(x, x1));
			}

			objFunc.line_search_end();

			// std::cout << cur_iter << " " << MAX_STEP_SIZE_ITER << " " << alpha << std::endl;

			if (alpha <= 1e-7)
				return std::nan("");
			else
				return alpha;
		}
	};
} // namespace polyfem
