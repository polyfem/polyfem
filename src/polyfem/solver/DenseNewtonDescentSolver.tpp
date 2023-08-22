#pragma once

#include "DenseNewtonDescentSolver.hpp"
// #include <unsupported/Eigen/SparseExtra>

namespace cppoptlib
{
	template <typename ProblemType>
	DenseNewtonDescentSolver<ProblemType>::DenseNewtonDescentSolver(
		const json &solver_params, const json &linear_solver_params, const double dt)
		: Superclass(solver_params, dt)
	{
	}

	// =======================================================================

	template <typename ProblemType>
	std::string DenseNewtonDescentSolver<ProblemType>::descent_strategy_name(int descent_strategy) const
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
	void DenseNewtonDescentSolver<ProblemType>::increase_descent_strategy()
	{
		if (this->descent_strategy == 0 || reg_weight > reg_weight_max)
			this->descent_strategy++;
		else
			reg_weight = std::max(reg_weight_inc * reg_weight, reg_weight_min);
		assert(this->descent_strategy <= 2);
	}

	// =======================================================================

	template <typename ProblemType>
	void DenseNewtonDescentSolver<ProblemType>::reset(const int ndof)
	{
		Superclass::reset(ndof);
		reg_weight = 0;
		internal_solver_info = json::array();
	}

	// =======================================================================

	template <typename ProblemType>
	bool DenseNewtonDescentSolver<ProblemType>::compute_update_direction(
		ProblemType &objFunc,
		const TVector &x,
		const TVector &grad,
		TVector &direction)
	{
		// if (this->descent_strategy == 2)
		// {
		// 	direction = -grad;
		// 	return true;
		// }

		Eigen::MatrixXd hessian;

		assemble_hessian(objFunc, x, hessian);

		if (this->descent_strategy == 2)
		{
			hessian.topLeftCorner(x.size(), x.size()) = Eigen::MatrixXd::Identity(x.size(), x.size());
		}
		if (!solve_linear_system(hessian, grad, direction))
			return compute_update_direction(objFunc, x, grad, direction);

		if (!check_direction(hessian, grad, direction))
			return compute_update_direction(objFunc, x, grad, direction);

		reg_weight /= reg_weight_dec;
		if (reg_weight < reg_weight_min)
			reg_weight = 0;

		return true;
	}

	// =======================================================================

	template <typename ProblemType>
	void DenseNewtonDescentSolver<ProblemType>::assemble_hessian(
		ProblemType &objFunc, const TVector &x, Eigen::MatrixXd &hessian)
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
			for (int k = 0; k < x.size(); k++)
				hessian(k, k) += reg_weight;
		}
	}

	// =======================================================================

	template <typename ProblemType>
	bool DenseNewtonDescentSolver<ProblemType>::solve_linear_system(
		const Eigen::MatrixXd &hessian, const TVector &grad, TVector &direction)
	{
		POLYFEM_SCOPED_TIMER("linear solve", this->inverting_time);

		TVector b = -grad;
		b.conservativeResize(hessian.rows());
		b.segment(grad.size(), b.size() - grad.size()).setZero();

		try
		{
			direction = hessian.ldlt().solve(b);
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
		
		const double residual = (hessian * direction - b).norm(); // H Δx + g = 0
		direction.conservativeResizeLike(grad);
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
	bool DenseNewtonDescentSolver<ProblemType>::check_direction(
		const Eigen::MatrixXd &hessian, const TVector &grad, const TVector &direction)
	{
		// do this check here because we need to repeat the solve without resetting reg_weight
		if (grad.dot(direction) >= 0)
		{
			increase_descent_strategy();
			polyfem::logger().log(
				log_level(), "[{}] direction is not a descent direction (Δx⋅g={}≥0); reverting to {}",
				name(), direction.dot(grad), descent_strategy_name());
			return false;
		}

		return true;
	}

	// =======================================================================

	template <typename ProblemType>
	void DenseNewtonDescentSolver<ProblemType>::update_solver_info(const double energy)
	{
		Superclass::update_solver_info(energy);
		this->solver_info["internal_solver"] = internal_solver_info;
	}

	// =======================================================================

	template <typename ProblemType>
	static bool has_hessian_nans(const Eigen::MatrixXd &hessian)
	{
		return std::isnan(hessian.norm());
	}
} // namespace cppoptlib
