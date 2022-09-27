#include "Remesh.hpp"

#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/mesh/remesh/MMGRemesh.hpp>
#include <polyfem/mesh/remesh/WildRemesh2D.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <igl/PI.h>
#include <igl/boundary_facets.h>

namespace polyfem::mesh
{
	void remesh(State &state, const double time, const double dt)
	{
		const int dim = state.mesh->dimension();
		Eigen::MatrixXd rest_positions;
		Eigen::MatrixXi elements;
		state.build_mesh_matrices(rest_positions, elements);

		assert(!state.mesh->is_volume());
		WildRemeshing2D remeshing(state.obstacle);
		remeshing.create_mesh(
			rest_positions,
			rest_positions + utils::unflatten(state.sol, dim),
			utils::unflatten(state.solve_data.time_integrator->v_prev(), dim),
			utils::unflatten(state.solve_data.time_integrator->a_prev(), dim),
			elements);

		for (int i = 0; i < 1; ++i)
		{
			// remeshing.smooth_all_vertices();
			// remeshing.split_all_edges();
			remeshing.collapse_all_edges();
		}

		state.load_mesh(remeshing.rest_positions(), remeshing.triangles());
		// FIXME:
		state.mesh->compute_boundary_ids(1e-6);
		state.mesh->set_body_ids(std::vector<int>(state.mesh->n_elements(), 1));
		state.set_materials(); // TODO: Explain why I need this?
		state.build_basis();
		state.assemble_rhs();
		state.assemble_stiffness_mat();

		state.sol = utils::flatten(remeshing.displacements());

		state.solve_data.rhs_assembler = state.build_rhs_assembler();
		state.init_nonlinear_tensor_solve(time);
		if (state.problem->is_time_dependent())
		{
			state.solve_data.time_integrator->init(
				state.sol, utils::flatten(remeshing.velocities()),
				utils::flatten(remeshing.accelerations()), dt);
		}
	}
} // namespace polyfem::mesh
