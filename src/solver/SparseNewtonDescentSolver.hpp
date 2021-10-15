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

			linear_solver = polysolve::LinearSolver::create(
				linear_solver_type, linear_precond_type);
			linear_solver->setParameters(this->solver_params);

			next_hessian = 0;
		}

		virtual void compute_update_direction(
			ProblemType &objFunc,
			const TVector &x,
			const TVector &grad,
			TVector &direction) override
		{
			if (this->use_gradient_descent)
			{
				direction = -grad;
				return;
			}

			new_hessian = this->m_current.iterations == next_hessian;
			if (new_hessian)
			{
				POLYFEM_SCOPED_TIMER("[timing] assembly time {}s", this->assembly_time);
				objFunc.hessian(x, hessian);

				++next_hessian;

				// if (this->m_current.iterations == 0 && has_hessian_nans(hessian))
				// {
				// 	this->m_status = Status::UserDefined;
				// 	polyfem::logger().debug("stopping because hessian is nan");
				// 	this->m_error_code = Superclass::ErrorCode::NanEncountered;
				// 	throw std::runtime_error("stopping because hessian is nan");
				// }
			}

			{
				POLYFEM_SCOPED_TIMER("[timing] linear solve {}s", this->inverting_time);
				if (new_hessian)
				{
					// TODO: get the correct size
					linear_solver->analyzePattern(hessian, hessian.rows());

					try
					{
						linear_solver->factorize(hessian);
					}
					catch (const std::runtime_error &err)
					{
						polyfem::logger().error("Unable to factorize Hessian: \"{}\"; reverting to gradient descent", err.what());
						// polyfem::write_sparse_matrix_csv("problematic_hessian.csv", hessian);
						this->use_gradient_descent = true;
						direction = -grad;
						return;
					}
				}
				linear_solver->solve(-grad, direction); // H Δx = -g
			}

			// gradient descent, check descent direction
			const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0
			if (std::isnan(residual) || residual > 1e-7)
			{
				polyfem::logger().warn("large linear solve residual ({}); reverting to gradient descent", residual);
				direction = -grad;
				this->use_gradient_descent = true;
			}
			else
			{
				polyfem::logger().trace("linear solve residual {}", residual);
			}

			if (grad.squaredNorm() != 0 && direction.dot(grad) >= 0)
			{
				polyfem::logger().warn("Newton direction is not a descent direction (Δx⋅g={}≥0); reverting to gradient descent", direction.dot(grad));
				direction = -grad;
				this->use_gradient_descent = true;
			}

			json info;
			linear_solver->getInfo(info);
			internal_solver_info.push_back(info);
		}

		void handle_small_step(double step) override
		{
			if (new_hessian)
			{
				if (this->use_gradient_descent)
				{
					polyfem::logger().warn(
						"[{}] (iter={}) ||step||={} is too small; stopping",
						name(), this->m_current.iterations, step);
					this->m_status = Status::UserDefined;
					this->m_error_code = Superclass::ErrorCode::StepTooSmall;
				}
				else
				{
					polyfem::logger().debug(
						"[{}] (iter={}) ||step||={} is too small; trying gradient descent",
						name(), this->m_current.iterations, step);
					this->use_gradient_descent = true;
				}
			}
			else
			{
				next_hessian = this->m_current.iterations;
				polyfem::logger().debug(
					"[{}] (iter={}) ||step||={} is too small; force recompute hessian",
					name(), this->m_current.iterations, step);
			}
		}

		void update_solver_info() override
		{
			Superclass::update_solver_info();

			this->solver_info["internal_solver"] = internal_solver_info;
			this->solver_info["internal_solver_first"] = internal_solver_info.front();
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
