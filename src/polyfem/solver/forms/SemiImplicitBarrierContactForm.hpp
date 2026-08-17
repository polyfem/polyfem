#pragma once

#include "BarrierContactForm.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>

namespace polyfem::solver
{
	/// Barrier contact with per-contact stiffness frozen for each Newton solve.
	class SemiImplicitBarrierContactForm final : public BarrierContactForm
	{
	public:
		SemiImplicitBarrierContactForm(
			const ipc::CollisionMesh &collision_mesh,
			const double dhat,
			const double avg_mass,
			const bool use_area_weighting,
			const bool use_improved_max_operator,
			const bool is_time_dependent,
			const bool enable_shape_derivatives,
			const ipc::BroadPhaseMethod broad_phase_method,
			const double ccd_tolerance,
			const int ccd_max_iterations,
			const json &options = json::object());

		void update_barrier_stiffness(
			const Eigen::VectorXd &x,
			const Eigen::MatrixXd &grad_energy) override;
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		void set_system_hessian_provider(
			const std::function<void(const Eigen::VectorXd &, StiffnessMatrix &)> &provider)
		{
			system_hessian_provider_ = provider;
		}

		void set_system_gradient_provider(
			const std::function<void(const Eigen::VectorXd &, Eigen::VectorXd &)> &provider)
		{
			system_gradient_provider_ = provider;
		}

		void begin_subsolve(const Eigen::VectorXd &x);
		bool restart_requested(const Eigen::VectorXd &x, const int iteration) const;
		void retune_on_stall(const Eigen::VectorXd &x, const double factor);
		int project_floor_pairs(const Eigen::VectorXd &x, Eigen::VectorXd &dir) const;
		json diagnostics(const Eigen::VectorXd &x) const;
		int frozen_floor_pair_count() const { return int(floor_active_keys_.size()); }

		/// Assign frozen per-contact stiffness to an independently built set.
		void assign_collision_stiffness(ipc::NormalCollisions &collision_set) const;

	protected:
		void update_collision_set(const Eigen::MatrixXd &displaced_surface) override;

	private:
		bool calibrate_trim(const Eigen::VectorXd &x);
		void refresh_stiffness(
			const Eigen::VectorXd &x,
			const bool run_trim_controller = true);
		void bump_trim(const double factor);
		double collapse_bump_factor(const double avg_d2) const;
		double collapse_severity(const double avg_d2, const double min_d2) const;
		std::array<long, 5> collision_key(
			const ipc::NormalCollisions &collisions, const size_t i) const;
		void freeze_floor_active_set(const Eigen::MatrixXd &displaced_surface);

		std::function<void(const Eigen::VectorXd &, StiffnessMatrix &)>
			system_hessian_provider_;
		std::function<void(const Eigen::VectorXd &, Eigen::VectorXd &)>
			system_gradient_provider_;
		Eigen::MatrixXd kappa_surface_;
		StiffnessMatrix kappa_hessian_;
		mutable std::map<std::array<long, 5>, double> kappa_cache_;
		std::set<std::array<long, 5>> floor_active_keys_;
		bool subsolve_had_contacts_ = false;
		int iters_since_refresh_ = 0;
		int iters_since_trim_ = 0;
		double kappa_cap_ = std::numeric_limits<double>::infinity();
		double kappa_median_ = 0.0;
		bool kappa_snapshot_had_contacts_ = false;
		double kappa_hessian_max_ = 0.0;

		int refresh_interval_ = 0;
		double trim_lower_ = 0.5;
		double trim_upper_ = 0.9;
		double trim_factor_ = 2.0;
		double trim_min_ = std::pow(2.0, -32);
		double trim_max_ = std::pow(2.0, 32);
		double kappa_min_ = 0.0;
		double kappa_spread_ = 1e4;
		double conditioning_cap_ = 1e3;
		int controller_interval_ = 30;
		double constraint_floor_ = 1e-4;
	};
} // namespace polyfem::solver
