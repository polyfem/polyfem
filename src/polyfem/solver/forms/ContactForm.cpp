#include "ContactForm.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/io/OBJWriter.hpp>

#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

#include <igl/writePLY.h>

namespace polyfem::solver
{
	ContactForm::ContactForm(const CollisionMesh &collision_mesh,
							 const double dhat,
							 const double avg_mass,
							 const bool use_adaptive_barrier_stiffness,
							 const bool is_time_dependent,
							 const bool enable_shape_derivatives,
							 const ipc::BroadPhaseMethod broad_phase_method,
							 const double ccd_tolerance,
							 const int ccd_max_iterations)
		: collision_mesh_(collision_mesh),
		  dhat_(dhat),
		  use_adaptive_barrier_stiffness_(use_adaptive_barrier_stiffness),
		  barrier_stiffness_(1.0),
		  avg_mass_(avg_mass),
		  is_time_dependent_(is_time_dependent),
		  enable_shape_derivatives_(enable_shape_derivatives),
		  broad_phase_method_(broad_phase_method),
		  broad_phase_(ipc::create_broad_phase(broad_phase_method)),
		  tight_inclusion_ccd_(ccd_tolerance, ccd_max_iterations)
	{
		assert(dhat_ > 0);
		assert(ccd_tolerance > 0);

		prev_distance_ = -1;
	}

	void ContactForm::init(const Eigen::VectorXd &x)
	{
		update_collision_set(compute_displaced_surface(x));
	}

	void ContactForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		update_collision_set(compute_displaced_surface(x));
	}

	Eigen::MatrixXd ContactForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return collision_mesh_.displace_vertices(utils::unflatten(x, collision_mesh_.dim()));
	}

	void ContactForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		update_collision_set(compute_displaced_surface(new_x));
	}

	double ContactForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// Extract surface only
		const Eigen::MatrixXd V0 = compute_displaced_surface(x0);
		const Eigen::MatrixXd V1 = compute_displaced_surface(x1);

		if (save_ccd_debug_meshes)
		{
			const Eigen::MatrixXi E = collision_mesh_.dim() == 2 ? Eigen::MatrixXi() : collision_mesh_.edges();
			const Eigen::MatrixXi &F = collision_mesh_.faces();
			igl::writePLY(resolve_output_path("debug_ccd_0.ply"), V0, F, E);
			igl::writePLY(resolve_output_path("debug_ccd_1.ply"), V1, F, E);
		}

		double max_step;
		if (use_cached_candidates_ && broad_phase_method_ != ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE)
			max_step = candidates_.compute_collision_free_stepsize(
				collision_mesh_, V0, V1, dmin_, tight_inclusion_ccd_);
		else
			max_step = ipc::compute_collision_free_stepsize(
				collision_mesh_, V0, V1, dmin_, broad_phase_.get(), tight_inclusion_ccd_);

		if (save_ccd_debug_meshes && ipc::has_intersections(collision_mesh_, (V1 - V0) * max_step + V0, broad_phase_.get()))
		{
			log_and_throw_error("Taking max_step results in intersections (max_step={})", max_step);
		}

#ifndef NDEBUG
		// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
		Eigen::MatrixXd V_toi = (V1 - V0) * max_step + V0;

		while (ipc::has_intersections(collision_mesh_, V_toi, broad_phase_.get()))
		{
			logger().error("Taking max_step results in intersections (max_step={:g})", max_step);
			max_step /= 2.0;

			const double Linf = (V_toi - V0).lpNorm<Eigen::Infinity>();
			if (max_step <= 0 || Linf == 0)
				log_and_throw_error("Unable to find an intersection free step size (max_step={:g} L∞={:g})", max_step, Linf);

			V_toi = (V1 - V0) * max_step + V0;
		}
#endif

		return max_step;
	}

	void ContactForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		candidates_.build(
			collision_mesh_,
			compute_displaced_surface(x0),
			compute_displaced_surface(x1),
			/*inflation_radius=*/barrier_support_size() / 2,
			broad_phase_.get());

		use_cached_candidates_ = true;
	}

	void ContactForm::line_search_end()
	{
		candidates_.clear();
		use_cached_candidates_ = false;
	}

	bool ContactForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		const auto displaced0 = compute_displaced_surface(x0);
		const auto displaced1 = compute_displaced_surface(x1);

		// Skip CCD if the displacement is zero.
		if ((displaced1 - displaced0).lpNorm<Eigen::Infinity>() == 0.0)
		{
			// Assumes initially intersection-free
			return true;
		}

		bool is_valid;
		if (use_cached_candidates_)
			is_valid = candidates_.is_step_collision_free(
				collision_mesh_, displaced0, displaced1, dmin_,
				tight_inclusion_ccd_);
		else
			is_valid = ipc::is_step_collision_free(
				collision_mesh_, displaced0, displaced1, dmin_, broad_phase_.get(),
				tight_inclusion_ccd_);

		return is_valid;
	}

	// Get the block variables *for the full problem* participating in the contact constraint.
	ContactForm::TrivialContactStencil ContactForm::trivialContactStencil(const std::array<int, 4> &vertex_ids) const {
		// Convert to block variables of the global system.
		TrivialContactStencil result(4);
		size_t back = 0;
		for (int vi : vertex_ids) {
			if (vi < 0) continue;
			int bvar = collision_mesh_.to_full_vertex_id(vi);
			result[back++] = bvar;
		}
		result.resize(back);
		return result;
	}

	// Get the block variables *for the full problem* participating in the contact constraint; nontrivial case.
	ContactForm::NontrivialContactStencil ContactForm::nontrivialContactStencil(const std::array<int, 4> &vertex_ids) const {
		NontrivialContactStencil result(MAX_CONTACT_STENCIL_SIZE);

		size_t back = 0;
		for (int vi : vertex_ids)
		{
			if (vi < 0)
				continue;

			collision_mesh_.visit_displacement_map_row(vi, [&](int bvar, double /* weight */) {
				if (back >= result.size())
					log_and_throw_error("Contact stencil exceeds maximum size of {}. Consider increasing MAX_CONTACT_STENCIL_SIZE.", MAX_CONTACT_STENCIL_SIZE);
				result[back] = bvar;
				++back;
			});
		}

		result.conservativeResize(back);
		return result;
	}

	// In the case of a nontrivial contact stencil involving weighted one-to-many
	// local-to-global mappings (e.g., when using a proxy mesh), apply a change
	// of variables so that the block rows/cols of H_e correspond directly to
	// the full non-deduplicated stencil and absorb all associated weights.
	// (This is analogous to `expand_hessian_to_raw_stencil` in Assembler.cpp.)
	void ContactForm::expandContactHessianToNontrivialStencil(
		const std::array<int, 4> &vertex_ids,
		Eigen::MatrixXd &H_e) const
	{
		const int dim = int(collision_mesh_.dim());

		std::array<int, 5> offsets;
		int ncv = 0; // number of collision mesh vertices participating in constraint (reduced stencil size)
		int raw_stencil_size = 0;

		NontrivialContactStencilWeights weights(MAX_CONTACT_STENCIL_SIZE);

		for (int vi : vertex_ids)
		{
			if (vi < 0) continue;

			offsets[ncv] = raw_stencil_size;
			collision_mesh_.visit_displacement_map_row(vi, [&](int /* bvar */, double weight) {
				if (raw_stencil_size >= weights.size()) log_and_throw_error("Contact stencil exceeds maximum size of {}. Consider increasing MAX_CONTACT_STENCIL_SIZE.", MAX_CONTACT_STENCIL_SIZE);
				weights[raw_stencil_size] = weight;
				++raw_stencil_size;
			});
			++ncv;
		}
		offsets[ncv] = raw_stencil_size;

		assert(H_e.rows() == ncv * dim);
		assert(H_e.cols() == ncv * dim);

		Eigen::MatrixXd reduced_H_e = std::move(H_e);
		H_e.resize(raw_stencil_size * dim, raw_stencil_size * dim);

		for (int j = 0; j < ncv; ++j)
		{
			for (int i = 0; i <= j; ++i)
			{
				const auto block = reduced_H_e.block(i * dim, j * dim, dim, dim);

				for (int ii = offsets[i]; ii < offsets[i + 1]; ++ii)
				{
					const int jj_begin = (i == j) ? ii : offsets[j];
					for (int jj = jj_begin; jj < offsets[j + 1]; ++jj)
						H_e.block(ii * dim, jj * dim, dim, dim) = weights[ii] * weights[jj] * block;
				}
			}
		}
	}

} // namespace polyfem::solver
