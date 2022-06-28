#pragma once

#include <polyfem/Common.hpp>
#include "NonlinearSolver.hpp"
#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

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
				if (reg_weight == 0)
					return "projected Newton";
				return fmt::format("projected Newton w/ regularization weight={}", reg_weight);
			case 2:
				return "gradient descent";
			default:
				throw std::invalid_argument("invalid descent strategy");
			}
		}

		void increase_descent_strategy() override
		{
			if (this->descent_strategy == 0 || reg_weight > reg_weight_max)
				this->descent_strategy++;
			else
				reg_weight = std::max(reg_weight_inc * reg_weight, reg_weight_min);
			assert(this->descent_strategy <= 2);
		}

	private:
		std::unique_ptr<polysolve::LinearSolver> linear_solver;
		const std::string linear_solver_type;
		const std::string linear_precond_type;

		json internal_solver_info = json::array();

		polyfem::StiffnessMatrix hessian; // Cached version of hessian

		// Regularization Coefficients
		double reg_weight = 0;
		static constexpr double reg_weight_min = 1e-8; // needs to be greater than zero
		static constexpr double reg_weight_max = 1e8;
		static constexpr double reg_weight_inc = 10;
		static constexpr double reg_weight_dec = 2;

		void reset(const ProblemType &objFunc, const TVector &x) override
		{
			Superclass::reset(objFunc, x);

			reg_weight = 0;

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
				POLYFEM_SCOPED_TIMER("assembly time", this->assembly_time);

				if (this->descent_strategy == 1)
					objFunc.set_project_to_psd(true);
				else if (this->descent_strategy == 0)
					objFunc.set_project_to_psd(false);
				else
					assert(false);

				objFunc.hessian(x, hessian);

				if (reg_weight > 0)
				{
					hessian += reg_weight * polyfem::utils::sparse_identity(hessian.rows(), hessian.cols());
				}
			}

			{
				POLYFEM_SCOPED_TIMER("linear solve", this->inverting_time);
				// TODO: get the correct size
				linear_solver->analyzePattern(hessian, hessian.rows());

				try
				{
					linear_solver->factorize(hessian);
				}
				catch (const std::runtime_error &err)
				{
					increase_descent_strategy();
					// warn if using gradient descent
					polyfem::logger().log(
						this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
						"Unable to factorize Hessian: \"{}\"; reverting to {}",
						err.what(), this->descent_strategy_name());
					// polyfem::write_sparse_matrix_csv("problematic_hessian.csv", hessian);
					return compute_update_direction(objFunc, x, grad, direction);
				}

				linear_solver->solve(-grad, direction); // H Δx = -g
			}

			// gradient descent, check descent direction
			const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
			if (std::isnan(residual))
			{
				increase_descent_strategy();
				polyfem::logger().log(
					this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
					"nan linear solve residual {} (||∇f||={}); reverting to {}",
					residual, grad.norm(), this->descent_strategy_name());
				return compute_update_direction(objFunc, x, grad, direction);
			}
			else if (residual > std::max(1e-8 * grad.norm(), 1e-5))
			{
				increase_descent_strategy();
				polyfem::logger().log(
					this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
					"large linear solve residual {} (||∇f||={}); reverting to {}",
					residual, grad.norm(), this->descent_strategy_name());
				return compute_update_direction(objFunc, x, grad, direction);
			}
			else
			{
				polyfem::logger().trace("linear solve residual {}", residual);
			}

			// do this check here because we need to repeat the solve without resetting reg_weight
			if (grad.dot(direction) >= 0)
			{
				increase_descent_strategy();
				polyfem::logger().log(
					this->descent_strategy == 2 ? spdlog::level::warn : spdlog::level::debug,
					"[{}] direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
					name(), direction.dot(grad), descent_strategy_name());
				return compute_update_direction(objFunc, x, grad, direction);
			}

			json info;
			linear_solver->getInfo(info);
			internal_solver_info.push_back(info);

			reg_weight /= reg_weight_dec;
			if (reg_weight < reg_weight_min)
				reg_weight = 0;

			return true;
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
