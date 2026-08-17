#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/lagrangian/AugmentedLagrangianForm.hpp>

#include <polysolve/nonlinear/Solver.hpp>

#include <Eigen/Core>

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace polyfem::solver
{
	enum class ALStrategy
	{
		Legacy,
		AdaptiveInexact,
	};

	NLOHMANN_JSON_SERIALIZE_ENUM(
		ALStrategy,
		{{ALStrategy::Legacy, "legacy"},
		 {ALStrategy::AdaptiveInexact, "adaptive_inexact"}})

	/// Options for contact-triggered nonlinear restarts.
	class StallRestartOptions
	{
	public:
		bool enabled = false;
		double alpha_threshold = 1e-2;
		int patience = 5;
		int min_iterations = 5;
		int soft_iteration_limit = -1;
		int max_restarts = 20;
	};

	/// Options for the conservative inexact-Newton AL strategy.
	class InexactALOptions
	{
	public:
		ALStrategy strategy = ALStrategy::Legacy;
		int inner_max_iterations = 20;
		int min_iterations = 5;
		int energy_window = 5;
		double min_relative_energy_decrease = 1e-6;
		double constraint_reduction_target = 0.5;
		int max_outer_iterations = 50;
		int max_consecutive_failures = 3;

		static InexactALOptions from_json(const json &params);
	};

	class ALSolver
	{
		using NLSolver = polysolve::nonlinear::Solver;

	public:
		ALSolver(
			const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &alagr_form,
			const double initial_al_weight,
			const double scaling,
			const double max_al_weight,
			const double eta_tol,
			const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness,
			const StallRestartOptions &stall_opts = StallRestartOptions(),
			const std::function<void(const Eigen::VectorXd &)> &on_stall = nullptr,
			const InexactALOptions &inexact_opts = InexactALOptions(),
			const std::function<bool(const Eigen::VectorXd &, int)> &contact_restart_requested = nullptr);
		virtual ~ALSolver() = default;

		void solve_al(NLProblem &nl_problem, Eigen::MatrixXd &sol,
					  std::shared_ptr<polysolve::nonlinear::Solver> nl_solver)
		{
			solve_al(nl_problem, sol, json{}, json{}, 1, nl_solver);
		}

		void solve_al(NLProblem &nl_problem, Eigen::MatrixXd &sol,
					  const json &nl_solver_params,
					  const json &linear_solver,
					  const double characteristic_length,
					  std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = nullptr);

		void solve_reduced(NLProblem &nl_problem, Eigen::MatrixXd &sol,
						   std::shared_ptr<polysolve::nonlinear::Solver> nl_solver)
		{
			solve_reduced(nl_problem, sol, json{}, json{}, 1, nl_solver);
		}

		void solve_reduced(NLProblem &nl_problem, Eigen::MatrixXd &sol,
						   const json &nl_solver_params,
						   const json &linear_solver,
						   const double characteristic_length,
						   std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = nullptr);

		std::function<void(const double)> post_subsolve = [](const double) {};

		/// Optional projection applied to every Newton direction.
		std::function<void(const Eigen::VectorXd &, Eigen::VectorXd &)> direction_filter = nullptr;

		/// Return and clear structured reports generated since the last call.
		json consume_subsolve_diagnostics();

	protected:
		enum class StopReason
		{
			Converged,
			IterationCap,
			EnergyStall,
			LineSearchStall,
			ContactStall,
			ContactRefresh,
			HardContactStall
		};

		class SubsolveResult
		{
		public:
			StopReason reason = StopReason::Converged;
			int iterations = 0;
			double initial_energy = std::numeric_limits<double>::quiet_NaN();
			double final_energy = std::numeric_limits<double>::quiet_NaN();
			double relative_energy_decrease = std::numeric_limits<double>::quiet_NaN();
			double last_alpha = std::numeric_limits<double>::quiet_NaN();
			double last_step = std::numeric_limits<double>::quiet_NaN();

			bool is_contact_stop() const
			{
				return reason == StopReason::ContactStall
					   || reason == StopReason::ContactRefresh
					   || reason == StopReason::HardContactStall;
			}
		};

		SubsolveResult minimize_once(
			NLProblem &nl_problem,
			Eigen::VectorXd &tmp_sol,
			const json &nl_solver_params,
			const json &linear_solver,
			const double characteristic_length,
			const std::shared_ptr<NLSolver> &nl_solverin,
			const bool apply_inexact_limits);

		void solve_al_legacy(
			NLProblem &nl_problem, Eigen::MatrixXd &sol,
			const json &nl_solver_params, const json &linear_solver,
			const double characteristic_length,
			const std::shared_ptr<NLSolver> &nl_solverin);

		void solve_al_inexact(
			NLProblem &nl_problem, Eigen::MatrixXd &sol,
			const json &nl_solver_params, const json &linear_solver,
			const double characteristic_length,
			const std::shared_ptr<NLSolver> &nl_solverin);

		bool projected_state_is_valid(
			NLProblem &nl_problem, const Eigen::VectorXd &full_sol,
			Eigen::VectorXd *projected_reduced = nullptr);

		double constraint_error(const Eigen::VectorXd &x) const;
		void record_subsolve(const SubsolveResult &result, const json &extra = json::object());
		static std::string stop_reason_name(const StopReason reason);

		std::vector<std::shared_ptr<AugmentedLagrangianForm>> alagr_forms;
		const double initial_al_weight;
		const double scaling;
		const double max_al_weight;
		const double eta_tol;

		std::function<void(const Eigen::VectorXd &)> update_barrier_stiffness;
		const StallRestartOptions stall_opts;
		std::function<void(const Eigen::VectorXd &)> on_stall;
		const InexactALOptions inexact_opts;
		std::function<bool(const Eigen::VectorXd &, int)> contact_restart_requested;
		json subsolve_diagnostics_ = json::array();
	};
} // namespace polyfem::solver
