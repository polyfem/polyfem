#pragma once

#include <polyfem/varforms/NonlinearElasticVarForm.hpp>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::assembler
{
	class Density;
	class GenericScalarProblem;
	class Laplacian;
	class MixedNLAssembler;
} // namespace polyfem::assembler

namespace polyfem::solver
{
	class AugmentedLagrangianForm;
	class Form;
	class GrowthDissipationForm;
	class MixedAssemblerForm;
	class NLProblem;
	class StackedForm;
} // namespace polyfem::solver

namespace polyfem::varform
{
	/// Variational form for the (displacement, growth) problem built on the
	/// GrowthElasticity mixed assembler. Structural clone of
	/// ThermoElasticVarForm with the transient machinery removed: Stage 1 of
	/// the growth build is static with the growth field theta prescribed
	/// (Dirichlet on the growth space); the growth block carries NO PDE of its
	/// own -- its stiffness comes from the K_gg quadrant of the growth
	/// correction plus the augmented-Lagrangian boundary forms, and with every
	/// growth DOF prescribed the reduced problem contains only displacements.
	///
	/// Deliberate deltas from the thermo precedent (each commented in the cpp):
	///  - static only: time-dependent configs are rejected with a clear error;
	///  - no growth-block ElasticForm / BodyForm / inertia / integrators;
	///  - the growth initial solution defaults to 1 (identity growth), because
	///    the zero default that is harmless for temperature is singular for
	///    theta (the assembler's positivity guard would throw on the first
	///    energy evaluation);
	///  - output fields "growth", "growth_gradient", and "growth_reaction" --
	///    the last is the growth-block gradient of the stacked energy at the
	///    converged full solution, i.e. the conjugate driving-force/reaction
	///    field of the formulation (the V3 deliverable).
	class GrowthElasticVarForm : public NonlinearElasticVarForm
	{
	public:
		using NonlinearElasticVarForm::NonlinearElasticVarForm;
		std::string name() const override { return "GrowthElastic"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) override;
		/// both blocks of the stacked (displacement, growth) solution, each with
		/// its node ordering, as for the other mixed formulations
		void serialize_checkpoint(io::CheckpointWriter &writer, const Eigen::MatrixXd &solution, const io::CheckpointMetadata &metadata) const override;
		void deserialize_checkpoint(const io::CheckpointReader &reader, Eigen::MatrixXd &solution) override;

		void export_data(const Eigen::MatrixXd &solution) const override;

		/// growth_law: first increment in which each growth DOF grew (0 = never);
		/// empty before a solved run. Also exported as growth_activation_step.
		const Eigen::VectorXi &growth_activation_step() const { return growth_activation_step_; }
		/// growth_law threshold route_b: f_gamma per element used by the last
		/// increment, and the threshold phi_e in use (empty / the given values otherwise).
		const Eigen::VectorXd &growth_shear_supply() const { return shear_supply_; }
		const Eigen::VectorXd &growth_threshold() const { return threshold_per_element_; }
		const Eigen::VectorXd &growth_threshold_parallel() const { return threshold_parallel_; }
		const Eigen::VectorXd &growth_attainment() const { return attainment_; }
		const Eigen::VectorXd &growth_lag_residual() const { return lag_residual_; }
		const Eigen::VectorXd &growth_dissipation() const { return dissipation_; }
		const Eigen::VectorXd &growth_shear_supply_work() const { return shear_work_; }

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	private:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;
		void solve_problem(
			Eigen::MatrixXd &sol,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step) override;
		void build_rhs_assembler() override;

		void read_material_space_ids(const json &args);
		json elastic_material_args() const;

		// ---- stress output of the grown material ----
		/// Grown-material stresses (GrowthElasticity::grown_stresses) at local
		/// points of element e, flattened column-major like the elastic tensor
		/// output: cauchy, pk1, pk2 (n x 9) and von Mises of the Cauchy stress (n x 1).
		void grown_stress_at(
			const int e, const Eigen::MatrixXd &local_pts, const double t,
			const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &growth,
			Eigen::MatrixXd &cauchy, Eigen::MatrixXd &pk1, Eigen::MatrixXd &pk2,
			Eigen::MatrixXd &von_mises) const;
		/// The elastic base evaluates von_mises, cauchy_stess_*, pk1_stess_*,
		/// pk2_stess_* (and the *_avg variants) with the ungrown material at the
		/// total F; overwrite the ones it emitted with the grown-material values.
		void replace_stress_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &growth,
			std::vector<io::OutputField> &fields) const;

		void build_displacement_boundary(mesh::Mesh &mesh);
		void build_growth_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args);
		void build_growth_boundary(mesh::Mesh &mesh);
		void build_forms(Eigen::MatrixXd &solution, const double t);
		/// NLProblem over the current forms and AL forms, initialized at sol, t.
		void make_nl_problem(const Eigen::MatrixXd &sol, const double t);
		void solve_nonlinear_step(const int step, Eigen::MatrixXd &solution, double grad_tol_factor = 1.0);

		// ---- Stage 2: solved growth (growth_law on the material) ----
		void read_growth_law(const json &args);
		void compute_growth_node_weights();
		/// w_a = sum_e vn_e phi_e int_e N_a dV from threshold_per_element_ (and the
		/// nodal volumes); called once (fixed threshold) or per update (route_b).
		void assemble_growth_node_weights();
		/// the threshold the state implies (route_b): G(x) = (PHI_PAR, f_gamma) at
		/// C_check of the state; f_gamma = f_I(Ce) - f_I(Ce with the transverse shears
		/// removed and Ce_nn at the layer value), PHI_PAR = max(f_par, 0)
		void threshold_target(const Eigen::MatrixXd &solution, double t,
							  Eigen::VectorXd &par, Eigen::VectorXd &gam, Eigen::VectorXd &clamped);
		/// install a threshold (parts), rebuild the node weights, log
		void set_threshold(const Eigen::VectorXd &par, const Eigen::VectorXd &gam,
						   const Eigen::VectorXd &clamped, double t, double omega);
		/// Route B at C_check of displacement u, per element: f_I(Ce) (phi_full),
		/// f_I(Ce shear-free, layer thickness) (phi_no_shear), and the total in-plane
		/// area stretch theta_tot. f_par = phi_no_shear, f_gamma = phi_full - phi_no_shear.
		void evaluate_threshold_state(const Eigen::VectorXd &u, double t,
									  Eigen::VectorXd &phi_full, Eigen::VectorXd &phi_no_shear, Eigen::VectorXd &theta_tot);
		/// element mean of a nodal growth field (growth quadrature)
		Eigen::VectorXd element_mean_growth(const Eigen::VectorXd &theta);
		/// lag residual phi(converged state) - phi used, and attainment, logged per increment
		void report_lag_residual(int step, double t, const Eigen::MatrixXd &solution);
		/// route-B split of the increment's dissipation term (D, S), accumulated and logged
		void accumulate_dissipation(int step, double t, const Eigen::VectorXd &dtheta);
		/// phi - f_I per v_n-weighted nodal volume on the settled set: free and pinned
		void log_kkt_residual(int step, double t, const Eigen::MatrixXd &solution,
							  const Eigen::VectorXd &mult, const std::vector<char> &pinned);
		static double median_of(const Eigen::VectorXd &v);
		void solve_growth_law(Eigen::MatrixXd &solution);
		/// one increment (t_from, t_to], split recursively on failure; returns passes
		int advance_increment(int step, double t_from, double t_to, Eigen::MatrixXd &solution, int depth);
		/// D/S accumulation, theta_n update and activation marking after a converged (sub)increment
		void commit_increment(int step, double t, Eigen::MatrixXd &solution);
		/// growth_law.save_increments: the converged stacked state of increment
		/// `step` (pseudo-time t) as growth_step_<step>.vtu, and growth_steps.pvd
		/// rewritten to list steps 1..step.
		void save_growth_increment(int step, double t, const Eigen::MatrixXd &solution);
		/// One increment: primal-dual active-set loop on theta >= theta_n.
		/// Returns the number of passes; throws if the set does not settle.
		int solve_growth_increment(const int step, const double t, Eigen::MatrixXd &solution);
		void solve_with_current_pins(const int step, const double t, Eigen::MatrixXd &solution, double grad_tol_factor = 1.0);
		/// d Pi_inc / d theta at the solution: stacked energy gradient plus the
		/// dissipation weights (the KKT residual; >= 0 where pinned, ~0 where free).
		Eigen::VectorXd growth_multiplier(const Eigen::MatrixXd &solution) const;

		void initial_growth_solution(Eigen::MatrixXd &solution) const;
		const Eigen::MatrixXd &prescribed_growth_field() const;
		void split_solution(
			const Eigen::MatrixXd &solution,
			Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &growth) const;
		Eigen::MatrixXd stacked_solution(
			const Eigen::MatrixXd &displacement,
			const Eigen::MatrixXd &growth) const;

		int displacement_ndof() const { return space_.ndof(); }
		int growth_ndof() const { return growth_space_.ndof(); }
		int total_ndof() const { return displacement_ndof() + growth_ndof(); }

		FESpace growth_space_;
		VarFormBoundaryState growth_boundary_;
		std::shared_ptr<assembler::Problem> growth_problem_;

		assembler::AssemblyValsCache growth_ass_vals_cache_;
		assembler::AssemblyValsCache growth_mass_ass_vals_cache_;

		std::shared_ptr<assembler::Assembler> growth_assembler_;
		std::shared_ptr<assembler::MixedNLAssembler> growthelastic_assembler_;
		std::shared_ptr<assembler::Mass> growth_mass_assembler_;
		std::shared_ptr<assembler::RhsAssembler> growth_rhs_assembler_;
		std::shared_ptr<assembler::Density> growth_rhs_density_;

		StiffnessMatrix growth_mass_;
		StiffnessMatrix stacked_lumped_mass_;
		Eigen::MatrixXd growth_rhs_;

		std::shared_ptr<solver::MixedAssemblerForm> growth_coupling_form_;
		std::shared_ptr<solver::StackedForm> stacked_form_;

		/// Converged stacked (u, growth) solution cached at the end of
		/// solve_problem. The output pipeline does not reliably hand
		/// output_fields the stacked vector (sizes/ordering differ per export
		/// path), so growth sampling and the reaction evaluation use this
		/// authoritative copy whenever the passed solution is not
		/// total_ndof-sized.
		Eigen::MatrixXd converged_solution_;

		/// Whole-field growth prescription (boundary_conditions.growth_nodal_field):
		/// nodal theta values loaded lazily from file; when set, EVERY growth DOF
		/// is Dirichlet with these targets (via BCLagrangianForm's tabulated-target
		/// constructor) and the same field is the growth initial solution.
		mutable Eigen::MatrixXd prescribed_growth_;

		int displacement_space_id_ = -1;
		int growth_space_id_ = -1;
		std::string elastic_formulation_ = "MaterialSum";

		/// Stage-2 growth law (material "growth_law"): rate-independent, per-element
		/// threshold (given, or route B from the state), irreversibility
		/// theta >= theta_n enforced by a primal-dual active set (pinned DOFs are
		/// Dirichlet at theta_n through BCLagrangianForm; free DOFs are unconstrained).
		struct GrowthLawSettings
		{
			bool enabled = false;
			int n_steps = 1;                   ///< pseudo-time increments over (0, t_end]
			double t_end = 1.0;                ///< pseudo-time at which DBC expressions reach their end state
			int max_active_set_iterations = 30;
			double active_set_tol = 1e-8;      ///< release a pin if dPi/dtheta < -tol * scale_a (Pa)
			double feasibility_tol = 1e-12;    ///< pin a free DOF if theta < theta_n - tol
			double theta_floor_fraction = 0.5; ///< positivity filter: theta >= fraction * theta_n
			bool save_increments = false;      ///< write growth_step_<k>.vtu + growth_steps.pvd per increment
			/// threshold "route_b": phi_e = max(f_par, 0) + f_gamma at C_check of the
			/// current displacement, built by the solver and re-evaluated after every
			/// active-set pass (Aitken-relaxed) until it moves by less than relag_tol,
			/// or at most relag_max_updates times per increment (0 = no cap; the first
			/// update is on the elastic predictor). Otherwise the threshold is the
			/// given number or file, fixed for the run.
			bool route_b = false;
			double theta_crit = 0;  ///< C_check in-plane elastic area stretch (route_b; else optional, lag report)
			double relag_tol = 1.0; ///< route_b: per-pass threshold iteration tolerance [Pa]
			int relag_max_updates = 0;
			/// < 1: after the active set settles, one confirmation pass with the
			/// nonlinear gradient tolerance multiplied by this factor (1 = off)
			double final_tol_factor = 1.0;
			/// a failed increment is retried from its start as two half increments,
			/// recursively, up to this many levels (0 = abort on the first failure)
			int max_increment_splits = 3;
		} growth_law_;

		Eigen::VectorXd threshold_per_element_;    ///< phi_e [Pa] in use (route_b: phi_par + f_gamma)
		Eigen::VectorXd threshold_parallel_;       ///< route_b: phi_par,e of the last update [Pa]
		Eigen::VectorXd shear_supply_;             ///< route_b: f_gamma,e of the last update [Pa]
		bool setup_logged_ = false;                ///< the setup line is logged once, not per update
		Eigen::VectorXd threshold_clamped_;        ///< 1 where f_par(C_check) < 0 (zero floor), when computed here
		Eigen::VectorXd lag_residual_;             ///< phi(converged state) - phi used, last increment [Pa]
		Eigen::VectorXd attainment_;               ///< vartheta^e / theta_crit at the last converged increment
		Eigen::VectorXd dissipation_;              ///< cumulative D = sum v_n max(f_par,0) dtheta per element [Pa]
		Eigen::VectorXd shear_work_;               ///< cumulative S = sum v_n f_gamma dtheta per element [Pa]
		Eigen::VectorXd element_volume_;           ///< reference element volumes (growth quadrature)
		Eigen::VectorXd normal_growth_per_element_; ///< vn_e (1 when not given)
		Eigen::VectorXd growth_node_volume_;       ///< int N_a dV
		Eigen::VectorXd growth_node_vn_volume_;    ///< int vn N_a dV (KKT residual in Pa)
		Eigen::VectorXd growth_node_weights_;      ///< int vn phi N_a dV
		Eigen::VectorXd theta_n_;                  ///< converged growth of the previous increment
		std::vector<int> pinned_growth_nodes_;     ///< member: BCLagrangianForm keeps a reference
		std::vector<char> growth_pinned_mask_;     ///< last solved pass: 1 = pinned at theta_n
		Eigen::VectorXi growth_activation_step_;  ///< first increment in which a growth DOF grew (0 = never)
		std::shared_ptr<solver::GrowthDissipationForm> dissipation_form_;
	};
} // namespace polyfem::varform
