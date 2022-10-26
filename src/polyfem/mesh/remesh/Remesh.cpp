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
	namespace
	{
		Eigen::MatrixXd combine_projection_quantaties(const State &state, const Eigen::MatrixXd &sol)
		{
			if (state.solve_data.time_integrator == nullptr)
				return Eigen::MatrixXd();

			const int ndof = state.mesh->n_vertices() * state.mesh->dimension();
			assert(sol.size() - ndof == state.obstacle.n_vertices() * state.mesh->dimension());

			// not including current displacement as this will be handled as positions
			Eigen::MatrixXd projection_quantities(ndof, 3 * state.solve_data.time_integrator->steps());
			int i = 0;
			for (const Eigen::VectorXd &x : state.solve_data.time_integrator->x_prevs())
				projection_quantities.col(i++) = x.head(ndof);
			for (const Eigen::VectorXd &v : state.solve_data.time_integrator->v_prevs())
				projection_quantities.col(i++) = v.head(ndof);
			for (const Eigen::VectorXd &a : state.solve_data.time_integrator->a_prevs())
				projection_quantities.col(i++) = a.head(ndof);
			assert(i == projection_quantities.cols());

			return projection_quantities;
		}

		void split_projection_quantaties(
			const State &state,
			const Eigen::MatrixXd &projected_quantities,
			std::vector<Eigen::VectorXd> &x_prevs,
			std::vector<Eigen::VectorXd> &v_prevs,
			std::vector<Eigen::VectorXd> &a_prevs)
		{
			if (state.solve_data.time_integrator == nullptr)
				return;

			const int n_vertices = state.mesh->n_vertices();
			const int dim = state.mesh->dimension();
			const int ndof_mesh = n_vertices * dim;
			const int ndof_obstacle = state.obstacle.n_vertices() * dim;

			const std::array<std::vector<Eigen::VectorXd> *, 3> all_prevs{{&x_prevs, &v_prevs, &a_prevs}};
			const int n_steps = state.solve_data.time_integrator->steps();

			int offset = 0;
			for (std::vector<Eigen::VectorXd> *prevs : all_prevs)
			{
				prevs->clear();
				for (int i = 0; i < n_steps; ++i)
				{
					prevs->emplace_back(ndof_mesh + ndof_obstacle);
					prevs->back().head(ndof_mesh) = utils::reorder_matrix(
						projected_quantities.col(offset + i), state.in_node_to_node, n_vertices, dim);
					// TODO: Set this to the correct previous position
					prevs->back().tail(ndof_obstacle).setZero();
				}
				offset += n_steps;
			}

			assert(offset == projected_quantities.cols());
		}
	} // namespace

	void remesh(State &state, Eigen::MatrixXd &sol, const double time, const double dt)
	{
		const int dim = state.mesh->dimension();
		Eigen::MatrixXd rest_positions;
		Eigen::MatrixXi elements;
		state.build_mesh_matrices(rest_positions, elements);

		WildRemeshing2D::EdgeMap edge_to_boundary_id;
		for (int ei = 0; ei < state.mesh->n_edges(); ei++)
		{
			int e0 = state.in_node_to_node[state.mesh->edge_vertex(ei, 0)];
			int e1 = state.in_node_to_node[state.mesh->edge_vertex(ei, 1)];
			if (e1 < e0)
				std::swap(e0, e1);
			edge_to_boundary_id[std::make_pair(e0, e1)] = state.mesh->get_boundary_id(ei);
		}

		const std::vector<int> &body_ids = state.mesh->get_body_ids();
		assert(body_ids.size() == elements.rows());

		// not including current displacement as this will be handled as positions
		const Eigen::MatrixXd projection_quantities = combine_projection_quantaties(state, sol);
		assert(projection_quantities.rows() == rest_positions.size());

		// Only remesh the FE mesh
		assert(sol.size() - rest_positions.size() == state.obstacle.n_vertices() * dim);
		const Eigen::MatrixXd mesh_sol = sol.topRows(rest_positions.size());
		const Eigen::MatrixXd obstacle_sol = sol.bottomRows(state.obstacle.n_vertices() * dim);
		const Eigen::MatrixXd positions = rest_positions + utils::unflatten(mesh_sol, dim);

		assert(!state.mesh->is_volume());
		WildRemeshing2D remeshing(state.assembler, state.formulation(), state.obstacle);
		remeshing.init(rest_positions, positions, elements, projection_quantities, edge_to_boundary_id, body_ids);

		for (int i = 0; i < 1; ++i)
		{
			remeshing.split_all_edges();
			// remeshing.consolidate_mesh();
			// remeshing.smooth_all_vertices();
			// remeshing.collapse_all_edges();
		}

		remeshing.consolidate_mesh();

		// --------------------------------------------------------------------
		// create new mesh

		rest_positions = remeshing.rest_positions();
		elements = remeshing.triangles();
		state.load_mesh(rest_positions, elements);

		// --------------------------------------------------------------------
		// set body ids

		state.mesh->set_body_ids(remeshing.body_ids());

		// --------------------------------------------------------------------
		// set boundary ids

		const WildRemeshing2D::EdgeMap remesh_boundary_ids = remeshing.boundary_ids();
		std::vector<int> boundary_ids(state.mesh->n_edges(), -1);
		for (int i = 0; i < state.mesh->n_edges(); i++)
		{
			int e0 = state.mesh->edge_vertex(i, 0);
			int e1 = state.mesh->edge_vertex(i, 1);
			if (e1 < e0)
				std::swap(e0, e1);
			boundary_ids[i] = remesh_boundary_ids.at(std::make_pair(e0, e1));
		}
		state.mesh->set_boundary_ids(boundary_ids);

		// --------------------------------------------------------------------

		// NOTE: We need to set the materials again because when it was called in
		// state.load_mesh() the body ids were not correct.
		state.set_materials();

		state.build_basis();
		state.assemble_rhs();
		state.assemble_stiffness_mat();

		// --------------------------------------------------------------------

		const int ndof_mesh = state.mesh->n_vertices() * dim;
		const int ndof_obstacle = state.obstacle.n_vertices() * dim;

		sol.resize(ndof_mesh + ndof_obstacle, 1);
		sol.topRows(ndof_mesh) = utils::flatten(utils::reorder_matrix(
			remeshing.displacements(), state.in_node_to_node));
		if (state.obstacle.n_vertices() > 0)
			sol.bottomRows(ndof_obstacle) = obstacle_sol;

		state.solve_data.rhs_assembler = state.build_rhs_assembler();
		state.init_nonlinear_tensor_solve(sol, time);

		if (state.problem->is_time_dependent())
		{
			std::vector<Eigen::VectorXd> x_prevs, v_prevs, a_prevs;
			split_projection_quantaties(
				state, remeshing.projected_quantities(), x_prevs, v_prevs, a_prevs);
			state.solve_data.time_integrator->init(x_prevs, v_prevs, a_prevs, dt);
		}

		// initialize the problem so contact force show up correctly in the output
		state.solve_data.nl_problem->init(sol);
		state.solve_data.updated_barrier_stiffness(sol);
	}
} // namespace polyfem::mesh
