#pragma once

#include "SparseNewtonDescentSolver.hpp"
#include <finitediff.hpp>
#include <unsupported/Eigen/SparseExtra>

namespace cppoptlib
{
	template <typename ProblemType>
	SparseNewtonDescentSolver<ProblemType>::SparseNewtonDescentSolver(
		const json &solver_params, const json &linear_solver_params, const double dt)
		: Superclass(solver_params, dt)
	{
		linear_solver = polysolve::LinearSolver::create(
			linear_solver_params["solver"], linear_solver_params["precond"]);
		linear_solver->setParameters(linear_solver_params);

		if (solver_params.contains("verify_hessian"))
			verify_hessian = solver_params["verify_hessian"];
	}

	// =======================================================================

	template <typename ProblemType>
	std::string SparseNewtonDescentSolver<ProblemType>::descent_strategy_name(int descent_strategy) const
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

	// =======================================================================

	template <typename ProblemType>
	void SparseNewtonDescentSolver<ProblemType>::increase_descent_strategy()
	{
		if (this->descent_strategy == 0 || reg_weight > reg_weight_max)
			this->descent_strategy++;
		else
			reg_weight = std::max(reg_weight_inc * reg_weight, reg_weight_min);
		assert(this->descent_strategy <= 2);
	}

	// =======================================================================

	template <typename ProblemType>
	void SparseNewtonDescentSolver<ProblemType>::reset(const int ndof)
	{
		Superclass::reset(ndof);
		assert(linear_solver != nullptr);
		reg_weight = 0;
		internal_solver_info = json::array();
	}

	// =======================================================================

	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::compute_update_direction(
		ProblemType &objFunc,
		const TVector &x,
		const TVector &grad,
		TVector &direction)
	{
		if (this->descent_strategy == 2)
		{
			direction = -grad;
			return true;
		}

		polyfem::StiffnessMatrix hessian;

		assemble_hessian(objFunc, x, hessian);

		if (!solve_linear_system(hessian, grad, direction))
			return compute_update_direction(objFunc, x, grad, direction);

		if (!check_direction(hessian, grad, direction))
			return compute_update_direction(objFunc, x, grad, direction);

		json info;
		linear_solver->getInfo(info);
		internal_solver_info.push_back(info);

		reg_weight /= reg_weight_dec;
		if (reg_weight < reg_weight_min)
			reg_weight = 0;

		return true;
	}

	// =======================================================================

	template <typename ProblemType>
	void SparseNewtonDescentSolver<ProblemType>::assemble_hessian(
		ProblemType &objFunc, const TVector &x, polyfem::StiffnessMatrix &hessian)
	{
		POLYFEM_SCOPED_TIMER("assembly time", this->assembly_time);

		if (this->descent_strategy == 1)
			objFunc.set_project_to_psd(true);
		else if (this->descent_strategy == 0)
			objFunc.set_project_to_psd(false);

		objFunc.hessian(x, hessian);

		if (verify_hessian)
		{
			Eigen::MatrixXd fhess;
			fd::finite_jacobian(
				x,
				[&](const Eigen::VectorXd &y) -> Eigen::VectorXd {
					Eigen::VectorXd grad;
					objFunc.solution_changed(y);
					objFunc.gradient(y, grad);
					return grad;
				},
				fhess);
			// fd::finite_hessian(
			// 	x,
			// 	[&](const Eigen::VectorXd &y) -> double {
			// 		Eigen::VectorXd grad;
			// 		objFunc.solution_changed(y);
			// 		return objFunc.value(y);
			// 	},
			// 	fhess);
			
			polyfem::StiffnessMatrix fhess_ = fhess.sparseView(0, 1e-9);
			const double error = (hessian - fhess_).norm();
			const double norm = hessian.norm();
			std::cout << "hessian FD error " << error << ", matrix norm " << norm << "\n";
			// if (error > 1e-5 * norm)
			// {
			// 	Eigen::saveMarket(hessian, "hess.mat");
			// 	Eigen::saveMarket(fhess_, "fhess.mat");
			// 	exit(0);
			// }
			objFunc.solution_changed(x);
		}

		if (reg_weight > 0)
		{
			hessian += reg_weight * polyfem::utils::sparse_identity(hessian.rows(), hessian.cols());
		}
	}

	// =======================================================================

	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::solve_linear_system(
		const polyfem::StiffnessMatrix &hessian, const TVector &grad, TVector &direction)
	{
		POLYFEM_SCOPED_TIMER("linear solve", this->inverting_time);
		// TODO: get the correct size
		linear_solver->analyzePattern(hessian, hessian.rows());

		TVector b = -grad;

		try
		{
			linear_solver->factorize(hessian);
		}
		catch (const std::runtime_error &err)
		{
			increase_descent_strategy();

			// warn if using gradient descent
			polyfem::logger().log(
				log_level(), "Unable to factorize Hessian: \"{}\"; reverting to {}",
				err.what(), this->descent_strategy_name());

			// polyfem::write_sparse_matrix_csv("problematic_hessian.csv", hessian);
			return false;
		}

		linear_solver->solve(b, direction); // H Δx = -g

		const double residual = (hessian * direction - b).norm(); // H Δx + g = 0
		if (std::isnan(residual) || residual > std::max(1e-8 * b.norm(), 1e-5))
		{
			increase_descent_strategy();

			polyfem::logger().log(
				log_level(),
				"[{}] large (or nan) linear solve residual {} (||∇f||={}); reverting to {}",
				name(), residual, grad.norm(), this->descent_strategy_name());

			return false;
		}
		else
		{
			polyfem::logger().trace("relative linear solve residual {}", residual / b.norm());
		}

		return true;
	}

	// =======================================================================

	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::check_direction(
		const polyfem::StiffnessMatrix &hessian, const TVector &grad, const TVector &direction)
	{
		// do this check here because we need to repeat the solve without resetting reg_weight
		if (grad.dot(direction) >= 0)
		{
			increase_descent_strategy();
			if (!this->disable_log)
				polyfem::logger().log(
					log_level(), "[{}] direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
					name(), direction.dot(grad), descent_strategy_name());
			return false;
		}

		return true;
	}

	// =======================================================================

	template <typename ProblemType>
	void SparseNewtonDescentSolver<ProblemType>::update_solver_info()
	{
		Superclass::update_solver_info();
		this->solver_info["internal_solver"] = internal_solver_info;
	}

	// =======================================================================

	template <typename ProblemType>
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
} // namespace cppoptlib
