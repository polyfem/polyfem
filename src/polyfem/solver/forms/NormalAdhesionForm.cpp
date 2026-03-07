#include "NormalAdhesionForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/Types.hpp>

#include <ipc/broad_phase/create_broad_phase.hpp>
#include <ipc/potentials/potential.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace polyfem::solver
{
	NormalAdhesionForm::NormalAdhesionForm(const ipc::CollisionMesh &collision_mesh,
										   const double dhat_p,
										   const double dhat_a,
										   const double Y,
										   const bool is_time_dependent,
										   const bool enable_shape_derivatives,
										   const ipc::BroadPhaseMethod broad_phase_method,
										   const double ccd_tolerance,
										   const int ccd_max_iterations)
		: collision_mesh_(collision_mesh),
		  dhat_p_(dhat_p),
		  dhat_a_(dhat_a),
		  Y_(Y),
		  is_time_dependent_(is_time_dependent),
		  enable_shape_derivatives_(enable_shape_derivatives),
		  broad_phase_method_(broad_phase_method),
		  broad_phase_(ipc::create_broad_phase(broad_phase_method)),
		  tight_inclusion_ccd_(ccd_tolerance, ccd_max_iterations),
		  normal_adhesion_potential_(dhat_p, dhat_a, Y, 1)
	{
		assert(dhat_p > 0);
		assert(dhat_a > dhat_p);
		assert(ccd_tolerance > 0);

		prev_distance_ = -1;
		collision_set_.set_enable_shape_derivatives(enable_shape_derivatives);
	}

	void NormalAdhesionForm::init(const Eigen::VectorXd &x)
	{
		update_collision_set(compute_displaced_surface(x));
	}

	void NormalAdhesionForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		update_collision_set(compute_displaced_surface(x));
	}

	Eigen::MatrixXd NormalAdhesionForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return collision_mesh_.displace_vertices(utils::unflatten(x, collision_mesh_.dim()));
	}

	void NormalAdhesionForm::update_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		// Store the previous value used to compute the constraint set to avoid duplicate computation.
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		if (use_cached_candidates_)
			collision_set_.build(
				candidates_, collision_mesh_, displaced_surface, dhat_a_);
		else
			collision_set_.build(
				collision_mesh_, displaced_surface, dhat_a_, dmin_, broad_phase_);
		cached_displaced_surface = displaced_surface;
	}

	double NormalAdhesionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return normal_adhesion_potential_(collision_set_, collision_mesh_, compute_displaced_surface(x));
	}

	Eigen::VectorXd NormalAdhesionForm::value_per_element_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd V = compute_displaced_surface(x);
		assert(V.rows() == collision_mesh_.num_vertices());

		const size_t num_vertices = collision_mesh_.num_vertices();

		if (collision_set_.empty())
		{
			return Eigen::VectorXd::Zero(collision_mesh_.full_num_vertices());
		}

		const Eigen::MatrixXi &E = collision_mesh_.edges();
		const Eigen::MatrixXi &F = collision_mesh_.faces();

		auto storage = utils::create_thread_storage<Eigen::VectorXd>(Eigen::VectorXd::Zero(num_vertices));

		utils::maybe_parallel_for(collision_set_.size(), [&](int start, int end, int thread_id) {
			Eigen::VectorXd &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (size_t i = start; i < end; i++)
			{
				// Quadrature weight is premultiplied by compute_potential
				const double potential = normal_adhesion_potential_(collision_set_[i], collision_set_[i].dof(V, E, F));

				const int n_v = collision_set_[i].num_vertices();
				const auto vis = collision_set_[i].vertex_ids(E, F);
				for (int j = 0; j < n_v; j++)
				{
					assert(0 <= vis[j] && vis[j] < num_vertices);
					local_storage[vis[j]] += potential / n_v;
				}
			}
		});

		Eigen::VectorXd out = Eigen::VectorXd::Zero(num_vertices);
		for (const auto &local_potential : storage)
		{
			out += local_potential;
		}

		Eigen::VectorXd out_full = Eigen::VectorXd::Zero(collision_mesh_.full_num_vertices());
		for (int i = 0; i < out.size(); i++)
			out_full[collision_mesh_.to_full_vertex_id(i)] = out[i];

		assert(std::abs(value_unweighted(x) - out_full.sum()) < std::max(1e-10 * out_full.sum(), 1e-10));

		return out_full;
	}

	void NormalAdhesionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = normal_adhesion_potential_.gradient(collision_set_, collision_mesh_, compute_displaced_surface(x));
		gradv = collision_mesh_.to_full_dof(gradv);
	}

	void NormalAdhesionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("normal adhesion hessian");

		ipc::PSDProjectionMethod psd_projection_method;

		if (project_to_psd_)
		{
			psd_projection_method = ipc::PSDProjectionMethod::CLAMP;
		}
		else
		{
			psd_projection_method = ipc::PSDProjectionMethod::NONE;
		}

		hessian = normal_adhesion_potential_.hessian(collision_set_, collision_mesh_, compute_displaced_surface(x), psd_projection_method);
		hessian = collision_mesh_.to_full_dof(hessian);
	}

	void NormalAdhesionForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		update_collision_set(compute_displaced_surface(new_x));
	}

	void NormalAdhesionForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		candidates_.build(
			collision_mesh_,
			compute_displaced_surface(x0),
			compute_displaced_surface(x1),
			/*inflation_radius=*/dhat_a_ / 2,
			broad_phase_);

		use_cached_candidates_ = true;
	}

	void NormalAdhesionForm::line_search_end()
	{
		candidates_.clear();
		use_cached_candidates_ = false;
	}

	void NormalAdhesionForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		if (data.iter_num == 0)
			return;

		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(data.x);

		const double curr_distance = collision_set_.compute_minimum_distance(collision_mesh_, displaced_surface);

		prev_distance_ = curr_distance;
	}

} // namespace polyfem::solver
