#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/NonlinearSolver.hpp>
#include <polysolve/LinearSolver.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

namespace cppoptlib
{

	template <typename ProblemType>
	class SparseNewtonDescentSolver : public NonlinearSolver<ProblemType>
	{
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		SparseNewtonDescentSolver(const json &solver_params, const std::string &linear_solver_type, const std::string &linear_precond_type)
			: Superclass(solver_params), linear_solver_type(linear_solver_type), linear_precond_type(linear_precond_type)
		{
		}

		std::string name() const override { return "SparseNewtonDescentSolver"; }

	private:
		std::unique_ptr<polysolve::LinearSolver> linear_solver;
		const std::string linear_solver_type;
		const std::string linear_precond_type;

		json internal_solver_info = json::array();

		polyfem::StiffnessMatrix hessian; // Cached version of hessian
		size_t next_hessian;              // Avoid recomputing hessian on line search failures
		bool new_hessian;

		void reset(ProblemType &objFunc, TVector &x) override
		{
			Superclass::reset(objFunc, x);

			igl::Timer timer;

			timer.start();
			linear_solver = polysolve::LinearSolver::create(linear_solver_type, linear_precond_type);
			linear_solver->setParameters(this->solver_params);
			timer.stop();
			polyfem::logger().debug("\tinternal solver {}, took {}s", linear_solver->name(), timer.getElapsedTimeInSec());

			next_hessian = 0;
		}

		virtual void compute_search_direction(ProblemType &objFunc, const TVector &x, const TVector &grad, TVector &direction)
		{
			igl::Timer timer;

			new_hessian = this->m_current.iterations == next_hessian;
			if (new_hessian && !this->line_search_failed)
			{
				timer.start();
				objFunc.hessian(x, hessian);
				timer.stop();
				polyfem::logger().debug("\tassembly time {}s", timer.getElapsedTimeInSec());
				this->assembly_time += timer.getElapsedTimeInSec();

				next_hessian += 1;

				// if (this->m_current.iterations == 0 && has_hessian_nans(hessian))
				// {
				// 	this->m_status = Status::UserDefined;
				// 	polyfem::logger().debug("stopping because hessian is nan");
				// 	this->m_error_code = -10;
				// 	throw std::runtime_error("stopping because hessian is nan");
				// }
			}

			timer.start();
			if (new_hessian && !this->line_search_failed)
			{
				//TODO: get the correct size
				linear_solver->analyzePattern(hessian, hessian.rows());
				linear_solver->factorize(hessian);
			}
			if (!this->line_search_failed)
			{
				linear_solver->solve(grad, direction);
			}

			//gradient descent, check descent direction
			const double residual = (hessian * direction - grad).norm();
			polyfem::logger().trace("residual {}", residual);
			if (this->line_search_failed || std::isnan(residual) || residual > 1e-7)
			{
				polyfem::logger().debug("\treverting to gradient descent, since residual is {}", residual);
				direction = grad;
			}

			direction *= -1; // Descent
			timer.stop();
			polyfem::logger().debug("\tinverting time {}s", timer.getElapsedTimeInSec());
			this->inverting_time += timer.getElapsedTimeInSec();

			json info;
			linear_solver->getInfo(info);
			internal_solver_info.push_back(info);
		}

		void handle_small_step(double step) override
		{
			if (new_hessian)
			{
				this->m_status = Status::UserDefined;
				polyfem::logger().debug("stopping because ||step||={} is too small", step);
				this->m_error_code = -1;
			}
			else
			{
				next_hessian = this->m_current.iterations;
				polyfem::logger().debug("\tstep small force recompute hessian");
			}
		}

		void update_solver_info() override
		{
			Superclass::update_solver_info();

			this->solver_info["internal_solver"] = internal_solver_info;
			this->solver_info["internal_solver_first"] = internal_solver_info.front();
		}

		bool has_hessian_nans(const polyfem::StiffnessMatrix &hessian)
		{
			for (int k = 0; k < hessian.outerSize(); ++k)
			{
				for (polyfem::StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
				{
					if (std::isnan(it.value()))
						return true;
				}
			}

			return false;
		}
	};
} // namespace cppoptlib
