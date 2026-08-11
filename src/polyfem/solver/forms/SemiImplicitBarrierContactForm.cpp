#include "SemiImplicitBarrierContactForm.hpp"

#include <polyfem/utils/Logger.hpp>

#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/barrier/barrier.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

namespace polyfem::solver
{
	namespace
	{
		class FlooredClampedLogBarrier : public ipc::ClampedLogBarrier
		{
		public:
			explicit FlooredClampedLogBarrier(const double floor_sq)
				: floor_sq_(floor_sq)
			{
			}

			double operator()(const double d, const double dhat) const override
			{
				if (d >= floor_sq_)
					return ipc::ClampedLogBarrier::operator()(d, dhat);
				return ipc::ClampedLogBarrier::operator()(floor_sq_, dhat)
					   + ipc::ClampedLogBarrier::first_derivative(floor_sq_, dhat)
							 * (d - floor_sq_);
			}

			double first_derivative(const double d, const double dhat) const override
			{
				return ipc::ClampedLogBarrier::first_derivative(
					std::max(d, floor_sq_), dhat);
			}

			double second_derivative(const double d, const double dhat) const override
			{
				return d < floor_sq_
					   ? 0.0
					   : ipc::ClampedLogBarrier::second_derivative(d, dhat);
			}

		private:
			double floor_sq_;
		};

		ipc::BarrierPotential make_barrier_potential(
			const double dhat, const json &options)
		{
			const double gap_floor =
				options.is_object() ? options.value("gap_floor", 0.0) : 0.0;
			if (gap_floor > 0)
			{
				return ipc::BarrierPotential(
					std::make_shared<FlooredClampedLogBarrier>(
						std::pow(gap_floor * dhat, 2)),
					dhat, 1.0, /*use_physical_barrier=*/false);
			}
			return ipc::BarrierPotential(
				dhat, 1.0, /*use_physical_barrier=*/false);
		}
	} // namespace

	SemiImplicitBarrierContactForm::SemiImplicitBarrierContactForm(
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
		const json &options)
		: BarrierContactForm(
			  collision_mesh, dhat, avg_mass,
			  use_area_weighting, use_improved_max_operator,
			  /*use_adaptive_barrier_stiffness=*/true,
			  is_time_dependent, enable_shape_derivatives,
			  broad_phase_method, ccd_tolerance, ccd_max_iterations,
			  make_barrier_potential(dhat, options))
	{
		if (enable_shape_derivatives)
			log_and_throw_error(
				"Semi-implicit barrier stiffness does not support shape derivatives!");

		if (options.is_object())
		{
			refresh_interval_ = options.value("refresh_interval", refresh_interval_);
			trim_lower_ = options.value("trim_lower", trim_lower_);
			trim_upper_ = options.value("trim_upper", trim_upper_);
			trim_factor_ = options.value("trim_factor", trim_factor_);
			trim_min_ = options.value("trim_min", trim_min_);
			trim_max_ = options.value("trim_max", trim_max_);
			kappa_min_ = options.value("kappa_min", kappa_min_);
			kappa_spread_ = options.value("kappa_spread", kappa_spread_);
			conditioning_cap_ = options.value("conditioning_cap", conditioning_cap_);
			controller_interval_ = options.value("controller_interval", controller_interval_);
			constraint_floor_ = options.value("constraint_floor", constraint_floor_);
		}

		refresh_interval_ = std::max(refresh_interval_, 0);
		kappa_min_ = std::max(kappa_min_, 0.0);
		if (!(trim_lower_ < trim_upper_))
			log_and_throw_error(
				"Semi-implicit barrier stiffness requires trim_lower < trim_upper!");
	}

	double SemiImplicitBarrierContactForm::collapse_severity(
		const double avg_d2, const double min_d2) const
	{
		constexpr double min_gap_slack = 1e2;
		double severity = std::numeric_limits<double>::infinity();
		if (std::isfinite(avg_d2))
			severity = avg_d2;
		if (std::isfinite(min_d2))
			severity = std::min(severity, min_d2 * min_gap_slack);
		return severity;
	}

	double SemiImplicitBarrierContactForm::collapse_bump_factor(
		const double avg_d2) const
	{
		return std::min(
			256.0,
			std::max(
				trim_factor_,
				std::sqrt(trim_lower_ * dhat_ * dhat_ / avg_d2)));
	}

	void SemiImplicitBarrierContactForm::bump_trim(const double factor)
	{
		const double new_trim =
			std::clamp(barrier_stiffness_ * factor, trim_min_, trim_max_);
		if (new_trim != barrier_stiffness_)
		{
			logger().debug(
				"Barrier stiffness trim: {:g} -> {:g}",
				barrier_stiffness_, new_trim);
			barrier_stiffness_ = new_trim;
			iters_since_trim_ = 0;
		}
	}

	int SemiImplicitBarrierContactForm::project_floor_pairs(
		const Eigen::VectorXd &x, Eigen::VectorXd &dir) const
	{
		if (!(constraint_floor_ > 0) || collision_set_.empty())
			return 0;

		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);
		const Eigen::MatrixXi &E = collision_mesh_.edges();
		const Eigen::MatrixXi &F = collision_mesh_.faces();
		const int dim = collision_mesh_.dim();

		std::vector<std::vector<std::pair<int, double>>> constraints;
		for (size_t i = 0; i < collision_set_.size(); i++)
		{
			const ipc::NormalCollision &collision = collision_set_[i];
			if (!floor_active_keys_.count(collision_key(collision_set_, i)))
				continue;

			const ipc::VectorMax12d positions = collision.dof(displaced_surface, E, F);
			const ipc::VectorMax12d local_grad =
				collision.compute_distance_gradient(positions);
			const auto vids = collision.vertex_ids(E, F);

			std::vector<std::pair<int, double>> entries;
			for (int a = 0; a < collision.num_vertices(); a++)
			{
				const long va = collision_mesh_.to_full_vertex_id(vids[a]);
				if (dim * va + dim > dir.size())
					continue;
				for (int k = 0; k < dim; k++)
					entries.emplace_back(
						dim * va + k, local_grad(dim * a + k));
			}
			if (!entries.empty())
				constraints.push_back(std::move(entries));
		}

		if (constraints.empty())
			return 0;

		constexpr int n_sweeps = 4;
		for (int sweep = 0; sweep < n_sweeps; sweep++)
		{
			bool any = false;
			for (const auto &g : constraints)
			{
				double dot = 0;
				double nrm2 = 0;
				for (const auto &[idx, val] : g)
				{
					dot += val * dir[idx];
					nrm2 += val * val;
				}
				if (dot < 0 && nrm2 > 0)
				{
					const double scale = dot / nrm2;
					for (const auto &[idx, val] : g)
						dir[idx] -= scale * val;
					any = true;
				}
			}
			if (!any)
				break;
		}

		logger().debug(
			"Constraint floor: projected closing components of {} pair(s) below {:g}*dhat",
			constraints.size(), constraint_floor_);
		return int(constraints.size());
	}

	void SemiImplicitBarrierContactForm::retune_on_stall(
		const Eigen::VectorXd &x, const double factor)
	{
		refresh_stiffness(x, /*run_trim_controller=*/false);
		if (collision_set_.empty())
			return;

		const double avg_d2 = collision_set_.compute_avg_distance(
			collision_mesh_, kappa_surface_, dhat_);
		const double min_d2 = collision_set_.compute_minimum_distance(
			collision_mesh_, kappa_surface_);
		const double severity = collapse_severity(avg_d2, min_d2);

		if (std::isfinite(severity)
			&& severity < trim_lower_ * dhat_ * dhat_)
		{
			bump_trim(std::max(factor, collapse_bump_factor(severity)));
		}
		else if (!calibrate_trim(x)
				 && std::isfinite(avg_d2)
				 && avg_d2 > trim_upper_ * dhat_ * dhat_)
		{
			bump_trim(1.0 / factor);
		}
		freeze_floor_active_set(kappa_surface_);
	}

	std::array<long, 5> SemiImplicitBarrierContactForm::collision_key(
		const ipc::NormalCollisions &collisions, const size_t i) const
	{
		const auto vids = collisions[i].vertex_ids(
			collision_mesh_.edges(), collision_mesh_.faces());
		long type_tag = 3;
		if (collisions.is_vertex_vertex(i))
			type_tag = 0;
		else if (collisions.is_edge_vertex(i))
			type_tag = 1;
		else if (collisions.is_edge_edge(i))
			type_tag = 2;
		else if (collisions.is_plane_vertex(i))
			type_tag = 4;
		return {
			{type_tag, long(vids[0]), long(vids[1]), long(vids[2]), long(vids[3])}};
	}

	void SemiImplicitBarrierContactForm::freeze_floor_active_set(
		const Eigen::MatrixXd &displaced_surface)
	{
		floor_active_keys_.clear();
		if (!(constraint_floor_ > 0))
			return;

		const double floor_d2 = std::pow(constraint_floor_ * dhat_, 2);
		const Eigen::MatrixXi &E = collision_mesh_.edges();
		const Eigen::MatrixXi &F = collision_mesh_.faces();
		for (size_t i = 0; i < collision_set_.size(); ++i)
		{
			const ipc::NormalCollision &collision = collision_set_[i];
			if (collision.compute_distance(
					collision.dof(displaced_surface, E, F))
				< floor_d2)
			{
				floor_active_keys_.insert(collision_key(collision_set_, i));
				collision_set_[i].stiffness_scale = 0.0;
			}
		}
	}

	void SemiImplicitBarrierContactForm::begin_subsolve(
		const Eigen::VectorXd &x)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);
		update_collision_set(displaced_surface);
		refresh_stiffness(x);
		freeze_floor_active_set(displaced_surface);
		subsolve_had_contacts_ = !collision_set_.empty();
		iters_since_refresh_ = 0;
		iters_since_trim_ = 0;
	}

	bool SemiImplicitBarrierContactForm::restart_requested(
		const Eigen::VectorXd &x, const int iteration) const
	{
		if (iteration <= 0)
			return false;
		if (!subsolve_had_contacts_ && !collision_set_.empty())
			return true;
		if (refresh_interval_ > 0
			&& iters_since_refresh_ >= refresh_interval_)
			return true;

		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);
		const Eigen::MatrixXi &E = collision_mesh_.edges();
		const Eigen::MatrixXi &F = collision_mesh_.faces();
		const double floor_d2 = std::pow(constraint_floor_ * dhat_, 2);
		for (size_t i = 0; i < collision_set_.size(); ++i)
		{
			const ipc::NormalCollision &collision = collision_set_[i];
			if (constraint_floor_ > 0
				&& collision.compute_distance(
					   collision.dof(displaced_surface, E, F))
					   < floor_d2
				&& !floor_active_keys_.count(collision_key(collision_set_, i)))
				return true;
		}

		if (collision_set_.empty())
			return false;
		const double avg_d2 = collision_set_.compute_avg_distance(
			collision_mesh_, displaced_surface, dhat_);
		const double min_d2 = collision_set_.compute_minimum_distance(
			collision_mesh_, displaced_surface);
		const double severity = collapse_severity(avg_d2, min_d2);
		if (iteration >= 3 && std::isfinite(severity)
			&& severity < trim_lower_ * dhat_ * dhat_)
			return true;
		return controller_interval_ > 0
			   && iteration >= controller_interval_
			   && std::isfinite(avg_d2)
			   && avg_d2 > trim_upper_ * dhat_ * dhat_;
	}

	json SemiImplicitBarrierContactForm::diagnostics(
		const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);
		const double avg_d2 = collision_set_.compute_avg_distance(
			collision_mesh_, displaced_surface, dhat_);
		const double min_d2 = collision_set_.compute_minimum_distance(
			collision_mesh_, displaced_surface);
		return {
			{"trim", barrier_stiffness_},
			{"contacts", collision_set_.size()},
			{"active_floor_pairs", floor_active_keys_.size()},
			{"average_gap_over_dhat",
			 std::isfinite(avg_d2) ? std::sqrt(avg_d2) / dhat_ : -1.0},
			{"minimum_gap_over_dhat",
			 std::isfinite(min_d2) ? std::sqrt(min_d2) / dhat_ : -1.0}};
	}

	void SemiImplicitBarrierContactForm::refresh_stiffness(
		const Eigen::VectorXd &x, const bool run_trim_controller)
	{
		if (!system_hessian_provider_)
			log_and_throw_error(
				"Semi-implicit barrier stiffness requires a system Hessian provider!");

		kappa_surface_ = compute_displaced_surface(x);
		system_hessian_provider_(x, kappa_hessian_);
		kappa_hessian_max_ = 0.0;
		for (int k = 0; k < kappa_hessian_.outerSize(); k++)
			for (StiffnessMatrix::InnerIterator it(kappa_hessian_, k); it; ++it)
				kappa_hessian_max_ =
					std::max(kappa_hessian_max_, std::abs(it.value()));
		kappa_cache_.clear();
		iters_since_refresh_ = 0;

		kappa_cap_ = std::numeric_limits<double>::infinity();
		kappa_median_ = 0.0;
		const bool first_contact =
			!kappa_snapshot_had_contacts_ && !collision_set_.empty();
		kappa_snapshot_had_contacts_ = !collision_set_.empty();
		assign_collision_stiffness(collision_set_);
		if (!collision_set_.empty())
		{
			std::vector<double> kappas(collision_set_.size());
			for (size_t i = 0; i < collision_set_.size(); i++)
				kappas[i] = collision_set_[i].stiffness_scale;
			std::nth_element(
				kappas.begin(), kappas.begin() + kappas.size() / 2, kappas.end());
			kappa_median_ = kappas[kappas.size() / 2];

			if (std::isfinite(kappa_spread_) && kappa_spread_ > 0)
			{
				kappa_cap_ = kappa_spread_ * kappa_median_;
				for (size_t i = 0; i < collision_set_.size(); i++)
				{
					collision_set_[i].stiffness_scale = std::min(
						collision_set_[i].stiffness_scale, kappa_cap_);
				}
			}
		}

		if (run_trim_controller && !collision_set_.empty())
		{
			const double avg_d2 = collision_set_.compute_avg_distance(
				collision_mesh_, kappa_surface_, dhat_);
			const double min_d2 = collision_set_.compute_minimum_distance(
				collision_mesh_, kappa_surface_);
			const double severity = collapse_severity(avg_d2, min_d2);
			const double dhat_sq = dhat_ * dhat_;

			if (std::isfinite(severity) && severity < trim_lower_ * dhat_sq)
			{
				bump_trim(collapse_bump_factor(severity));
			}
			else if (!calibrate_trim(x))
			{
				if (first_contact && kappa_median_ > 0
					&& kappa_hessian_max_ > 0
					&& !(std::isfinite(severity)
						 && severity < trim_lower_ * dhat_sq))
				{
					const double cap = std::clamp(
						conditioning_cap_ * kappa_hessian_max_
							/ (weight_ * kappa_median_),
						trim_min_, trim_max_);
					if (barrier_stiffness_ > cap)
					{
						logger().debug(
							"Conditioning cap on first contact: trim {:g} -> {:g}",
							barrier_stiffness_, cap);
						barrier_stiffness_ = cap;
						iters_since_trim_ = 0;
					}
				}
				if (std::isfinite(avg_d2)
					&& avg_d2 > trim_upper_ * dhat_ * dhat_)
					bump_trim(1.0 / trim_factor_);
			}
		}

		if (!collision_set_.empty())
		{
			double min_kappa = std::numeric_limits<double>::infinity();
			double max_kappa = 0;
			double mean_kappa = 0;
			for (size_t i = 0; i < collision_set_.size(); i++)
			{
				const double kappa = collision_set_[i].stiffness_scale;
				min_kappa = std::min(min_kappa, kappa);
				max_kappa = std::max(max_kappa, kappa);
				mean_kappa += kappa;
			}
			mean_kappa /= collision_set_.size();
			logger().debug(
				"Refreshed semi-implicit barrier stiffness over {} contacts: min={:g} mean={:g} max={:g} (trim={:g})",
				collision_set_.size(), min_kappa, mean_kappa, max_kappa,
				barrier_stiffness_);
		}
	}

	void SemiImplicitBarrierContactForm::assign_collision_stiffness(
		ipc::NormalCollisions &collision_set) const
	{
		if (kappa_surface_.size() == 0)
			return;

		const Eigen::MatrixXi &E = collision_mesh_.edges();
		const Eigen::MatrixXi &F = collision_mesh_.faces();
		const int dim = collision_mesh_.dim();

		for (size_t i = 0; i < collision_set.size(); i++)
		{
			if (collision_set.is_plane_vertex(i))
				continue;

			ipc::NormalCollision &collision = collision_set[i];
			const int n_verts = collision.num_vertices();
			const auto vids = collision.vertex_ids(E, F);
			const std::array<long, 5> key = collision_key(collision_set, i);

			double kappa;
			const auto cached = kappa_cache_.find(key);
			if (cached != kappa_cache_.end())
			{
				kappa = cached->second;
			}
			else
			{
				const ipc::VectorMax12d positions =
					collision.dof(kappa_surface_, E, F);
				// Do not freeze IPC's gap-dependent mass feasibility term. Dynamic
				// curvature is already represented by the inertia Hessian included in
				// kappa_hessian_; see the semi-implicit writeup, section 5.2.
				ipc::VectorMax4d local_mass = ipc::VectorMax4d::Zero(n_verts);
				ipc::MatrixMax12d local_hess = ipc::MatrixMax12d::Zero(
					dim * n_verts, dim * n_verts);

				for (int a = 0; a < n_verts; a++)
				{
					const long va = collision_mesh_.to_full_vertex_id(vids[a]);
					for (int b = 0; b < n_verts; b++)
					{
						const long vb =
							collision_mesh_.to_full_vertex_id(vids[b]);
						if (dim * va + dim > kappa_hessian_.rows()
							|| dim * vb + dim > kappa_hessian_.cols())
							continue;
						for (int k = 0; k < dim; k++)
							for (int l = 0; l < dim; l++)
							{
								local_hess(dim * a + k, dim * b + l) =
									kappa_hessian_.coeff(
										dim * va + k, dim * vb + l);
							}
					}
				}

				kappa = ipc::semi_implicit_stiffness(
					collision, positions, local_mass, local_hess, dmin_);
				kappa /= dhat_ * dhat_;
				if (!std::isfinite(kappa))
					kappa = 1e30;
				kappa = std::max(kappa, kappa_min_);
				kappa /= weight_;
				kappa_cache_.emplace(key, kappa);
			}

			collision.stiffness_scale = std::min(kappa, kappa_cap_);
		}
	}

	bool SemiImplicitBarrierContactForm::calibrate_trim(
		const Eigen::VectorXd &x)
	{
		if (!system_gradient_provider_ || collision_set_.empty())
			return false;

		Eigen::VectorXd grad_energy;
		system_gradient_provider_(x, grad_energy);
		Eigen::VectorXd grad_barrier = barrier_potential_.gradient(
			collision_set_, collision_mesh_, compute_displaced_surface(x));
		grad_barrier = collision_mesh_.to_full_dof(grad_barrier);

		const double barrier_norm = grad_barrier.norm();
		const double energy_norm = grad_energy.norm();
		if (!(barrier_norm > 0) || !(energy_norm > 0))
			return false;

		const double cos_opposition =
			-grad_barrier.dot(grad_energy) / (barrier_norm * energy_norm);
		constexpr double min_opposition = 0.1;
		if (!std::isfinite(cos_opposition)
			|| cos_opposition < min_opposition)
			return false;

		const double balanced_trim =
			cos_opposition * energy_norm / (weight_ * barrier_norm);
		if (!std::isfinite(balanced_trim) || balanced_trim <= 0)
			return false;

		const double new_trim =
			std::clamp(balanced_trim, trim_min_, trim_max_);
		if (new_trim > barrier_stiffness_)
		{
			logger().debug(
				"Gradient-balance trim calibration: {:g} -> {:g}",
				barrier_stiffness_, new_trim);
			barrier_stiffness_ = new_trim;
			iters_since_trim_ = 0;
		}
		return true;
	}

	void SemiImplicitBarrierContactForm::update_barrier_stiffness(
		const Eigen::VectorXd &x, const Eigen::MatrixXd &)
	{
		begin_subsolve(x);
	}

	void SemiImplicitBarrierContactForm::update_collision_set(
		const Eigen::MatrixXd &displaced_surface)
	{
		BarrierContactForm::update_collision_set(displaced_surface);
		assign_collision_stiffness(collision_set_);
		for (size_t i = 0; i < collision_set_.size(); ++i)
		{
			if (floor_active_keys_.count(collision_key(collision_set_, i)))
				collision_set_[i].stiffness_scale = 0.0;
		}
	}

	void SemiImplicitBarrierContactForm::post_step(
		const polysolve::nonlinear::PostStepData &data)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(data.x);
		const double min_d2 = collision_set_.compute_minimum_distance(
			collision_mesh_, displaced_surface);
		if (!std::isinf(min_d2))
		{
			const double ratio = std::sqrt(min_d2) / dhat();
			const auto log_level =
				ratio < 1e-6 ? spdlog::level::err
							 : (ratio < 1e-4 ? spdlog::level::warn
											   : spdlog::level::debug);
			logger().log(
				log_level, "Minimum distance during solve: {}, dhat: {}",
				std::sqrt(min_d2), dhat());
		}

		const double avg_d2 = collision_set_.compute_avg_distance(
			collision_mesh_, displaced_surface, dhat_);
		++iters_since_refresh_;
		++iters_since_trim_;
		logger().debug(
			"Frozen semi-implicit contact: trim={:g}, sqrt(avg d2)/dhat={:g}, sqrt(min d2)/dhat={:g}, floor_pairs={}",
			barrier_stiffness(),
			std::isfinite(avg_d2) ? std::sqrt(avg_d2) / dhat_ : -1.0,
			std::isfinite(min_d2) ? std::sqrt(min_d2) / dhat_ : -1.0,
			floor_active_keys_.size());
	}
} // namespace polyfem::solver
