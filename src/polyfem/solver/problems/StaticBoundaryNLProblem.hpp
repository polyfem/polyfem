#pragma once

#include <polyfem/solver/NLProblem.hpp>

namespace polyfem::solver
{
	class StaticBoundaryNLProblem : public NLProblem
	{
	public:
		StaticBoundaryNLProblem(
			const int full_size,
			const Eigen::VectorXd &boundary_values,
			const std::vector<std::shared_ptr<polyfem::solver::Form>> &forms,
			const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms)
			: polyfem::solver::NLProblem(full_size, forms, penalty_forms),
			  boundary_values_(boundary_values)
		{
		}

	protected:
		Eigen::MatrixXd constraint_values(const TVector &) const override { return boundary_values_; }

	private:
		const Eigen::MatrixXd boundary_values_;
	};
} // namespace polyfem::solver