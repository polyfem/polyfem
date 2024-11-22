#pragma once

#include "NLProblem.hpp"

namespace polyfem
{
	class State;
	namespace assembler
	{
		class MacroStrainValue;
	}
} // namespace polyfem

namespace polyfem::solver
{
	class NLHomoProblem : public NLProblem
	{
	public:
		using typename FullNLProblem::Scalar;
		using typename FullNLProblem::THessian;
		using typename FullNLProblem::TVector;

		NLHomoProblem(const int full_size,
					  const assembler::MacroStrainValue &macro_strain_constraint,
					  const State &state,
					  const double t,
					  const std::vector<std::shared_ptr<Form>> &forms,
					  const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms,
					  const bool solve_symmetric_macro_strain);
		virtual ~NLHomoProblem() = default;

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;
		void hessian(const TVector &x, THessian &hessian) override;

		void full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const override;

		int macro_reduced_size() const;

		TVector full_to_reduced(const TVector &full, const Eigen::MatrixXd &disp_grad) const;
		TVector full_to_reduced(const TVector &full) const override;
		TVector full_to_reduced_grad(const TVector &full) const override;
		TVector reduced_to_full(const TVector &reduced) const override;
		TVector reduced_to_full_shape_derivative(const Eigen::MatrixXd &disp_grad, const TVector &adjoint_full) const;

		TVector reduced_to_extended(const TVector &reduced, bool homogeneous = false) const;
		TVector extended_to_reduced(const TVector &extended) const;
		TVector extended_to_reduced_grad(const TVector &extended) const;
		void extended_hessian_to_reduced_hessian(const THessian &extended, THessian &reduced) const;

		Eigen::MatrixXd reduced_to_disp_grad(const TVector &reduced, bool homogeneous = false) const;

		void set_fixed_entry(const Eigen::VectorXi &fixed_entry);

		void init(const TVector &x0) override;
		bool is_step_valid(const TVector &x0, const TVector &x1) override;
		bool is_step_collision_free(const TVector &x0, const TVector &x1) override;
		double max_step_size(const TVector &x0, const TVector &x1) override;

		void line_search_begin(const TVector &x0, const TVector &x1) override;
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		void solution_changed(const TVector &new_x) override;

		void init_lagging(const TVector &x) override;
		void update_lagging(const TVector &x, const int iter_num) override;

		void update_quantities(const double t, const TVector &x) override;

		void add_form(const std::shared_ptr<Form> &form) { homo_forms.push_back(form); }
		bool has_symmetry_constraint() const { return only_symmetric; }

	protected:
		Eigen::MatrixXd constraint_values(const TVector &) const override;

	private:
		void init_projection();
		Eigen::MatrixXd constraint_grad() const;

		TVector macro_full_to_reduced(const TVector &full) const;
		Eigen::MatrixXd macro_full_to_reduced_grad(const Eigen::MatrixXd &full) const;
		TVector macro_reduced_to_full(const TVector &reduced, bool homogeneous = false) const;

		const State &state_;
		const bool only_symmetric;
		const assembler::MacroStrainValue &macro_strain_constraint_;

		Eigen::VectorXi fixed_mask_;
		Eigen::MatrixXd macro_mid_to_reduced_; // (dim*dim) x (dim*(dim+1)/2)
		Eigen::MatrixXd macro_full_to_mid_, macro_mid_to_full_;

		std::vector<std::shared_ptr<Form>> homo_forms;
	};
} // namespace polyfem::solver