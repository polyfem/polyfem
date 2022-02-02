#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/NonlinearSolver.hpp>
#include <polysolve/LinearSolver.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <polyfem/Logger.hpp>
#include <polyfem/Timer.hpp>

namespace cppoptlib
{
	template <typename ProblemType>
	class SparseNewtonDescentSolver : public NonlinearSolver<ProblemType>
	{
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		SparseNewtonDescentSolver(
			const json &solver_params,
			const std::string &linear_solver_type,
			const std::string &linear_precond_type)
			: Superclass(solver_params),
			  linear_solver_type(linear_solver_type),
			  linear_precond_type(linear_precond_type)
		{
		}

		std::string name() const override { return "Newton"; }

	protected:
		virtual int default_descent_strategy() override { return 0; }
		using Superclass::descent_strategy_name;
		std::string descent_strategy_name(int descent_strategy) const override
		{
			switch (descent_strategy)
			{
			case 0:
				return "Newton";
			case 1:
				return "projected Newton";
			case 2:
				return "gradient descent";
			default:
				throw "invalid descent strategy";
			}
		}

	private:
		std::unique_ptr<polysolve::LinearSolver> linear_solver;
		const std::string linear_solver_type;
		const std::string linear_precond_type;

		json internal_solver_info = json::array();

		polyfem::StiffnessMatrix hessian; // Cached version of hessian

		void reset(const ProblemType &objFunc, const TVector &x) override
		{
			Superclass::reset(objFunc, x);

			linear_solver = polysolve::LinearSolver::create(linear_solver_type, linear_precond_type);
			linear_solver->setParameters(this->solver_params);
		}

		virtual bool compute_update_direction(
			ProblemType &objFunc,
			const TVector &x,
			const TVector &grad,
			TVector &direction) override
		{
			if (this->descent_strategy == 2)
			{
				direction = -grad;
				return true;
			}

			{
				POLYFEM_SCOPED_TIMER("[timing] assembly time {}s", this->assembly_time);

				if (this->descent_strategy == 1)
					objFunc.set_project_to_psd(true);
				else if (this->descent_strategy == 0)
					objFunc.set_project_to_psd(false);
				else
					assert(false);

				objFunc.hessian(x, hessian);
			}

			{
				POLYFEM_SCOPED_TIMER("[timing] linear solve {}s", this->inverting_time);
				// TODO: get the correct size
				linear_solver->analyzePattern(hessian, hessian.rows());

				try
				{
					linear_solver->factorize(hessian);
				}
				catch (const std::runtime_error &err)
				{
					this->descent_strategy++;
					// Warning if we switch to projected Newton, else error
					polyfem::logger().log(
						this->descent_strategy == 1 ? spdlog::level::warn : spdlog::level::err,
						"Unable to factorize Hessian: \"{}\"; reverting to {}",
						err.what(), this->descent_strategy_name());
					// polyfem::write_sparse_matrix_csv("problematic_hessian.csv", hessian);
					return false;
				}

				linear_solver->solve(-grad, direction); // H Δx = -g
			}

			// gradient descent, check descent direction
			const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
			if (std::isnan(residual))                                    // || residual > 1e-7)
			{
				this->descent_strategy++;
				polyfem::logger().log(
					this->descent_strategy == 1 ? spdlog::level::warn : spdlog::level::err,
					"nan linear solve residual {} (||∇f||={}); reverting to {}",
					residual, grad.norm(), this->descent_strategy_name());
				return false;
			}
			else if (residual > 1e-5)
			{
				polyfem::logger().warn("large linear solve residual {} (||∇f||={})", residual, grad.norm());
			}
			else
			{
				polyfem::logger().trace("linear solve residual {}", residual);
			}

			if (grad.squaredNorm() != 0 && direction.dot(grad) >= 0)
			{
				this->descent_strategy++;
				polyfem::logger().log(
					this->descent_strategy == 1 ? spdlog::level::warn : spdlog::level::err,
					"Newton direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
					direction.dot(grad), this->descent_strategy_name());
				return false;
			}

			json info;
			linear_solver->getInfo(info);
			internal_solver_info.push_back(info);

			return true;
		}

		void handle_small_step(double step) override
		{
			if (this->descent_strategy == 0) //try to project to psd
				this->descent_strategy = 1;
			// if (this->use_gradient_descent)
			// {
			// 	// How did this not converge then?
			// 	// polyfem::logger().error(
			// 	// 	"[{}] (iter={}) ||step||={} is too small; stopping",
			// 	// 	name(), this->m_current.iterations, step);
			// 	// this->m_status = Status::UserDefined;
			// 	// this->m_error_code = Superclass::ErrorCode::StepTooSmall;
			// }
			// else
			// {
			// 	// Switching to gradient descent in this case will ruin quadratic convergence so don't.
			// 	// polyfem::logger().warn(
			// 	// 	"[{}] (iter={}) ||step||={} is too small; trying gradient descent",
			// 	// 	name(), this->m_current.iterations, step);
			// 	// this->use_gradient_descent = true;
			// }
		}

		void update_solver_info() override
		{
			Superclass::update_solver_info();

			this->solver_info["internal_solver"] = internal_solver_info;
			this->solver_info["internal_solver_first"] = internal_solver_info.size() ? internal_solver_info.front() : json(nullptr);
		}

		static bool has_hessian_nans(const polyfem::StiffnessMatrix &hessian)
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
