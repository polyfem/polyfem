#pragma once

#include "NLProblem.hpp"
#include "forms/PeriodicContactForm.hpp"

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
				  const double t, const std::vector<std::shared_ptr<Form>> &forms, 
				  const std::shared_ptr<PeriodicContactForm> &contact_form);
		
		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;
		void hessian(const TVector &x, THessian &hessian) override;

		void full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const override;

		TVector full_to_reduced(const TVector &full, const Eigen::MatrixXd &disp_grad) const;
		TVector full_to_reduced_grad(const TVector &full) const override;
		TVector reduced_to_full(const TVector &reduced) const override;

		int macro_reduced_size() const;
		TVector macro_full_to_reduced(const TVector &full) const;
		Eigen::MatrixXd macro_full_to_reduced_grad(const Eigen::MatrixXd &full) const;
		TVector macro_reduced_to_full(const TVector &reduced) const;

		TVector reduced_to_extended(const TVector &reduced) const;
		TVector extended_to_reduced_grad(const TVector &extended) const;
		void extended_hessian_to_reduced_hessian(const THessian &extended, THessian &reduced) const;

		Eigen::MatrixXd reduced_to_disp_grad(const TVector &reduced) const;

		void set_only_symmetric();
		void set_fixed_entry(const std::vector<int> &fixed_entry);


		bool is_step_valid(const TVector &x0, const TVector &x1) const override;
		bool is_step_collision_free(const TVector &x0, const TVector &x1) const override;
		double max_step_size(const TVector &x0, const TVector &x1) const override;

		void line_search_begin(const TVector &x0, const TVector &x1) override;
		void post_step(const int iter_num, const TVector &x) override;

		void solution_changed(const TVector &new_x) override;

		void init_lagging(const TVector &x) override;
		void update_lagging(const TVector &x, const int iter_num) override;

		void update_quantities(const double t, const TVector &x) override;

	private:
		Eigen::MatrixXd constraint_grad; // (dim*dim) x (dim*n_bases)

		bool only_symmetric = false;
		Eigen::MatrixXd full_to_symmetric, symmetric_to_full; // (dim*dim) x (dim*(dim+1)/2)

		std::vector<int> fixed_entry_; // dirichlet BC on macro strain

		std::shared_ptr<PeriodicContactForm> contact_form_;
    };
}