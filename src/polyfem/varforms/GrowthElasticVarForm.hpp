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
		std::string name() const override { return "GrowthElastic"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;

		io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) override;

		void export_data(const Eigen::MatrixXd &solution) const override;

		/// growth_law: first increment in which each growth DOF grew (0 = never);
		/// empty before a solved run. Also exported as growth_activation_step.
		const Eigen::VectorXi &growth_activation_step() const { return growth_activation_step_; }
		/// growth_law shear_supply "lagged": f_gamma per element used by the last
		/// increment, and the threshold phi_e in use (empty / file values otherwise).
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
		/// nodal volumes); called once (frozen) or per increment (lagged).
		void assemble_growth_node_weights();
		/// shear_supply = "lagged": f_gamma per element on the displacement of
		/// `solution` (C_check at theta_crit; f_gamma = f_I(Ce) - f_I(Ce with the
		/// transverse shears removed and Ce_nn at the layer value)), then
		/// threshold = phi_par + f_gamma and the node weights.
		void update_lagged_threshold(const Eigen::MatrixXd &solution, double t);
		/// the threshold is re-evaluated from the state during the run
		bool threshold_follows_state() const
		{
			return growth_law_.shear_supply == "lagged" || growth_law_.parallel_threshold == "lagged";
		}
		/// the threshold the state implies: G(x) = (PHI_PAR, f_gamma) at C_check of the
		/// state (PHI_PAR only when parallel_threshold is lagged, else the current one)
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

		/// Stage-2 growth law (material "growth_law"): rate-independent, constant
		/// per-element threshold within the run, irreversibility theta >= theta_n
		/// enforced by a primal-dual active set (pinned DOFs are Dirichlet at
		/// theta_n through BCLagrangianForm; free DOFs are unconstrained). The
		/// threshold is not yet lagged per increment (variant (b) is Stage 3).
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
			/// "frozen": phi_e is the threshold file's field for the whole run (route B
			/// at the end state's C_check). "lagged": only the dissipative part is
			/// frozen, phi_e = phi_par,e + f_gamma,e, with the shear supply f_gamma
			/// re-evaluated at the start of every increment on the previous
			/// increment's converged displacement (C_check construction at theta_crit).
			std::string shear_supply = "frozen";
			double theta_crit = 0;             ///< lagged: threshold in-plane elastic area stretch
			std::string parallel_field = "PHI_PAR"; ///< lagged: threshold-file field holding max(f_par(C_check), 0)
			/// lagged: when f_gamma is re-evaluated. "increment": once, on the previous
			/// increment's converged state. "predictor": on the elastic predictor of the
			/// increment (pass 1, growth pinned: the displacement at this increment's t),
			/// at no extra solve. "pass": after every active-set pass, iterating until
			/// the threshold changes by less than shear_update_tol (Pa).
			std::string shear_update = "increment";
			double shear_update_tol = 1.0;
			/// Dissipative part PHI_PAR of the route-B threshold. "file": the threshold
			/// file's parallel_field (V0b). "state": computed here at C_check of the
			/// displacement in threshold_state (default input.data.state), with the
			/// solver's own energy and fibers, then frozen; with shear_supply frozen,
			/// f_gamma is taken from the same state. "lagged": re-evaluated with
			/// f_gamma on the state (timing: shear_update), i.e. variant (b) frozen at
			/// the current state; with shear_update "pass" this is the in-increment
			/// re-lag (Stage 3, Q14).
			std::string parallel_threshold = "file";
			std::string threshold_state;       ///< parallel_threshold "state": state file with "u"
			bool lag_report = true;            ///< log the lag residual per increment (needs theta_crit)
			/// < 1: after the active set settles, one confirmation pass with the
			/// nonlinear gradient tolerance multiplied by this factor (1 = off)
			double final_tol_factor = 1.0;
			/// shear_update "pass": "aitken" relaxes the per-pass threshold iteration
			/// (Aitken / Irons-Tuck), omega clamped to [relag_omega_min, relag_omega_max]
			std::string relag_acceleration = "none";
			double relag_omega_min = 0.05;
			double relag_omega_max = 5.0;
			/// a failed increment is retried from its start as two half increments,
			/// recursively, up to this many levels (0 = abort on the first failure)
			int max_increment_splits = 3;
		} growth_law_;

		Eigen::VectorXd threshold_per_element_;    ///< phi_e [Pa] in use (lagged: phi_par + f_gamma)
		Eigen::VectorXd threshold_parallel_;       ///< lagged: phi_par,e, frozen [Pa]
		Eigen::VectorXd shear_supply_;             ///< lagged: f_gamma,e of the current increment [Pa]
		bool shear_supply_logged_once_ = false;    ///< the setup line is logged once, not per increment
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
