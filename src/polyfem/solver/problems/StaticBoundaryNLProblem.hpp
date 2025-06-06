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
			: polyfem::solver::NLProblem(full_size, forms, penalty_forms, nullptr),
			  boundary_values_(boundary_values)
		{
			throw std::runtime_error("To be fixed");
		}
		// TODO fix AL nullptr

	private:
		const Eigen::MatrixXd boundary_values_;
	};
} // namespace polyfem::solver