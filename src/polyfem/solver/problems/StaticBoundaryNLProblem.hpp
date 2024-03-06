#pragma once

#include <polyfem/solver/NLProblem.hpp>

namespace polyfem::solver
{
	class StaticBoundaryNLProblem : public NLProblem
	{
	public:
		StaticBoundaryNLProblem(
			const int full_size,
			const std::vector<int> &boundary_nodes,
			const Eigen::VectorXd &boundary_values,
			const std::vector<std::shared_ptr<polyfem::solver::Form>> &forms)
			: polyfem::solver::NLProblem(full_size, boundary_nodes, forms),
			  boundary_values_(boundary_values)
		{
		}

	protected:
		Eigen::MatrixXd boundary_values() const override { return boundary_values_; }

	private:
		const Eigen::MatrixXd boundary_values_;
	};
} // namespace polyfem::solver