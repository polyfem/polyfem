#pragma once

#include <polyfem/solver/NLProblem.hpp>

namespace polyfem::solver
{
    class NLHomoProblem : public NLProblem
    {
    public:
		using typename FullNLProblem::Scalar;
		using typename FullNLProblem::THessian;
		using typename FullNLProblem::TVector;

        NLHomoProblem(const int full_size,
				  const std::vector<int> &boundary_nodes,
				  const std::vector<mesh::LocalBoundary> &local_boundary,
				  const int n_boundary_samples,
				  const assembler::RhsAssembler &rhs_assembler,
				  const State &state,
				  const double t, const std::vector<std::shared_ptr<Form>> &forms);

		void full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const override;

		TVector full_to_reduced(const TVector &full, const Eigen::MatrixXd &disp_grad) const;
		TVector full_to_reduced_grad(const TVector &full) const override;
		TVector reduced_to_full(const TVector &reduced) const override;

		int macro_reduced_size() const;
		TVector macro_full_to_reduced(const TVector &full) const;
		Eigen::MatrixXd macro_full_to_reduced_grad(const Eigen::MatrixXd &full) const;
		TVector macro_reduced_to_full(const TVector &reduced) const;

		Eigen::MatrixXd reduced_to_disp_grad(const TVector &reduced) const;

		void set_only_symmetric();
		void set_fixed_entry(const std::vector<int> &fixed_entry);

	private:
		Eigen::MatrixXd constraint_grad; // (dim*dim) x (dim*n_bases)

		bool only_symmetric = false;
		Eigen::MatrixXd full_to_symmetric, symmetric_to_full; // (dim*dim) x (dim*(dim+1)/2)

		std::vector<int> fixed_entry_;
    };
}