#pragma once

#include "NLProblem.hpp"

namespace polyfem
{
	namespace assembler
	{
		class MacroStrainValue;
	}
	namespace mesh
	{
		class MeshNodes;
	}
} // namespace polyfem

namespace polyfem::solver
{

	/// For homogenization problem, u = ũ + GX
	/// u := displacement
	/// ũ := periodic fluctuation
	/// G := macro strain tensor, represent global change.
	///
	/// This class maps between three states: full, extended, and reduced.
	/// full := full nodal displacement u.
	/// extended := [ ũ | flatten G ]
	/// reduced := [ ũ | unknown component of flatten G ]
	///
	/// Note that user might choose to fix some component of G thus extended != reduced.
	class NLHomoProblem : public NLProblem
	{
	public:
		using typename FullNLProblem::Scalar;
		using typename FullNLProblem::THessian;
		using typename FullNLProblem::TVector;

		NLHomoProblem(const int full_size,
					  const assembler::MacroStrainValue &macro_strain_constraint,
					  int n_bases,
					  std::shared_ptr<mesh::MeshNodes> mesh_nodes,
					  double t,
					  const std::vector<std::shared_ptr<Form>> &forms,
					  const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms,
					  bool solve_symmetric_macro_strain,
					  const std::shared_ptr<polysolve::linear::Solver> &solver,
					  double char_length,
					  double char_force,
					  StiffnessMatrix lumped_mass,
					  int dimension);
		virtual ~NLHomoProblem() = default;

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;
		void hessian(const TVector &x, THessian &hessian) override;

		void full_hessian_to_reduced_hessian(THessian &hessian) const;

		int macro_reduced_size() const;

		TVector full_to_reduced(const TVector &full, const Eigen::MatrixXd &disp_grad) const;
		TVector full_to_reduced(const TVector &full) const;
		TVector full_to_reduced_grad(const TVector &full) const override;
		TVector full_to_reduced_diag(const TVector &full_diag) const override;
		TVector reduced_to_full(const TVector &reduced) const;

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

	private:
		void init_projection();
		Eigen::MatrixXd constraint_grad() const;

		TVector macro_full_to_reduced(const TVector &full) const;
		Eigen::MatrixXd macro_full_to_reduced_grad(const Eigen::MatrixXd &full) const;
		TVector macro_reduced_to_full(const TVector &reduced, bool homogeneous = false) const;

		const int n_bases_;
		const int dimension_;
		std::shared_ptr<mesh::MeshNodes> mesh_nodes_;
		const bool only_symmetric;
		const assembler::MacroStrainValue &macro_strain_constraint_;

		Eigen::VectorXi fixed_mask_;
		/// Selection matrix that maps potentially symmetric macro strain G to its free dof (reduced).
		/// Effective only if user fix some component of G.
		Eigen::MatrixXd macro_mid_to_reduced_; // (dim*dim) x (dim*(dim+1)/2)
		/// Selection matrix that maps full macro strain G to its upper triangular dof (mid).
		/// Enable if symmtric G is enable else a dummy.
		Eigen::MatrixXd macro_full_to_mid_;
		Eigen::MatrixXd macro_mid_to_full_;

		std::vector<std::shared_ptr<Form>> homo_forms;
	};
} // namespace polyfem::solver
