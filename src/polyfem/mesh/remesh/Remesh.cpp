#include "Remesh.hpp"

#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/mesh/remesh/MMGRemesh.hpp>
#include <polyfem/mesh/remesh/WildRemeshing2D.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <igl/PI.h>
#include <igl/boundary_facets.h>

namespace polyfem::mesh
{
	namespace
	{
		Eigen::MatrixXd combine_projection_quantities(const State &state, const Eigen::MatrixXd &sol)
		{
			if (state.solve_data.time_integrator == nullptr)
				return Eigen::MatrixXd();

			// not including current displacement as this will be handled as positions
			Eigen::MatrixXd projection_quantities(
				state.solve_data.time_integrator->x_prev().size(),
				3 * state.solve_data.time_integrator->steps());
			int i = 0;
			for (const Eigen::VectorXd &x : state.solve_data.time_integrator->x_prevs())
				projection_quantities.col(i++) = x;
			for (const Eigen::VectorXd &v : state.solve_data.time_integrator->v_prevs())
				projection_quantities.col(i++) = v;
			for (const Eigen::VectorXd &a : state.solve_data.time_integrator->a_prevs())
				projection_quantities.col(i++) = a;
			assert(i == projection_quantities.cols());

			return projection_quantities;
		}

		void split_projection_quantities(
			const State &state,
			const Eigen::MatrixXd &projected_quantities,
			std::vector<Eigen::VectorXd> &x_prevs,
			std::vector<Eigen::VectorXd> &v_prevs,
			std::vector<Eigen::VectorXd> &a_prevs)
		{
			if (projected_quantities.size() == 0)
				return;

			const int n_vertices = state.mesh->n_vertices();
			const int dim = state.mesh->dimension();
			const int ndof = state.n_bases * dim;
			const int ndof_mesh = n_vertices * dim;
			const int ndof_obstacle = state.obstacle.n_vertices() * dim;
			assert(projected_quantities.rows() == ndof);

			const std::array<std::vector<Eigen::VectorXd> *, 3> all_prevs{{&x_prevs, &v_prevs, &a_prevs}};
			const int n_steps = projected_quantities.cols() / 3;
			assert(projected_quantities.cols() % 3 == 0);

			int offset = 0;
			for (std::vector<Eigen::VectorXd> *prevs : all_prevs)
			{
				prevs->clear();
				for (int i = 0; i < n_steps; ++i)
				{
					prevs->push_back(projected_quantities.col(offset + i));
					prevs->back().head(ndof_mesh) = utils::reorder_matrix(
						prevs->back().head(ndof_mesh), state.in_node_to_node, n_vertices, dim);
				}
				offset += n_steps;
			}

			assert(offset == projected_quantities.cols());
		}
	} // namespace

	bool remesh(State &state, Eigen::MatrixXd &sol, const double time, const double dt)
	{
		const int dim = state.mesh->dimension();
		int ndof = sol.size();
		assert(sol.cols() == 1);
		int ndof_mesh = state.mesh->n_vertices() * dim;
		int ndof_obstacle = state.obstacle.n_vertices() * dim;
		assert(ndof == ndof_mesh + ndof_obstacle);

		Eigen::MatrixXd rest_positions;
		Eigen::MatrixXi elements;
		state.build_mesh_matrices(rest_positions, elements);
		assert(rest_positions.size() == ndof_mesh);

		WildRemeshing2D::EdgeMap<int> edge_to_boundary_id;
		for (int ei = 0; ei < state.mesh->n_edges(); ei++)
		{
			size_t e0 = state.in_node_to_node[state.mesh->edge_vertex(ei, 0)];
			size_t e1 = state.in_node_to_node[state.mesh->edge_vertex(ei, 1)];
			if (e1 < e0)
				std::swap(e0, e1);
			edge_to_boundary_id[std::make_pair(e0, e1)] = state.mesh->get_boundary_id(ei);
		}

		const std::vector<int> body_ids = state.mesh->has_body_ids() ? state.mesh->get_body_ids() : std::vector<int>(elements.rows(), 0);
		assert(body_ids.size() == elements.rows());

		// Only remesh the FE mesh
		assert(sol.size() - rest_positions.size() == state.obstacle.n_vertices() * dim);
		const Eigen::MatrixXd mesh_sol = sol.topRows(ndof_mesh);
		const Eigen::MatrixXd obstacle_sol = sol.bottomRows(ndof_obstacle);
		const Eigen::MatrixXd positions = rest_positions + utils::unflatten(mesh_sol, dim);

		// not including current displacement as this will be handled as positions
		Eigen::MatrixXd projection_quantities = combine_projection_quantities(state, sol);
		assert(projection_quantities.rows() == ndof);

		Eigen::VectorXd friction_gradient;
		if (state.solve_data.friction_form)
			state.solve_data.friction_form->first_derivative(sol, friction_gradient);
		else
			friction_gradient = Eigen::VectorXd::Zero(sol.size());
		projection_quantities.conservativeResize(Eigen::NoChange, projection_quantities.cols() + 1);
		assert(friction_gradient.size() == projection_quantities.rows());
		projection_quantities.rightCols(1) = friction_gradient;

		const Eigen::MatrixXd obstacle_projection_quantities = projection_quantities.bottomRows(ndof_obstacle);
		projection_quantities.conservativeResize(ndof_mesh, Eigen::NoChange);

		assert(!state.mesh->is_volume());
		WildRemeshing2D remeshing(state, utils::unflatten(obstacle_sol, dim), obstacle_projection_quantities, time, state.solve_data.nl_problem->value(sol));
		remeshing.energy_relative_tolerance = state.args["space"]["remesh"]["rel_tol"];
		remeshing.energy_absolute_tolerance = state.args["space"]["remesh"]["abs_tol"];
		remeshing.n_ring_size = state.args["space"]["remesh"]["n_ring_size"];
		remeshing.flood_fill_rel_area = state.args["space"]["remesh"]["flood_fill_rel_area"];
		remeshing.init(rest_positions, positions, elements, projection_quantities, edge_to_boundary_id, body_ids);

		const bool made_change = remeshing.execute(
			/*split=*/true, /*collapse=*/false, /*smooth=*/false, /*swap=*/false,
			/*max_ops_percent=*/-1);
		remeshing.timings.log();
		if (!made_change)
			return false;

		remeshing.consolidate_mesh();

		// --------------------------------------------------------------------
		// create new mesh

		// NOTE: Assumes only split ops were performed
		if (remeshing.rest_positions().rows() == state.mesh->n_vertices())
			return false;

		state.mesh = mesh::Mesh::create(remeshing.rest_positions(), remeshing.elements(), /*non_conforming=*/false);

		// set body ids
		state.mesh->set_body_ids(remeshing.body_ids());

		// set boundary ids
		const WildRemeshing2D::EdgeMap<int> remesh_boundary_ids = std::get<WildRemeshing2D::EdgeMap<int>>(remeshing.boundary_ids());
		std::vector<int> boundary_ids(state.mesh->n_edges(), -1);
		for (int i = 0; i < state.mesh->n_edges(); i++)
		{
			size_t e0 = state.mesh->edge_vertex(i, 0);
			size_t e1 = state.mesh->edge_vertex(i, 1);
			if (e1 < e0)
				std::swap(e0, e1);
			boundary_ids[i] = remesh_boundary_ids.at(std::make_pair(e0, e1));
		}
		state.mesh->set_boundary_ids(boundary_ids);

		// load mesh (and set materials) (will also reload obstacles from disk)
		state.load_mesh();

		// --------------------------------------------------------------------

		state.build_basis();
		state.assemble_rhs();
		state.assemble_stiffness_mat();

		// --------------------------------------------------------------------

		const int old_ndof = ndof;
		const int old_ndof_mesh = ndof_mesh;
		const int old_ndof_obstacle = ndof_obstacle;

		ndof_mesh = state.mesh->n_vertices() * dim;
		ndof_obstacle = state.obstacle.n_vertices() * dim;
		assert(ndof_obstacle == old_ndof_obstacle);
		ndof = state.n_bases * dim;
		assert(ndof == ndof_mesh + ndof_obstacle);

		sol.resize(ndof, 1);
		sol.topRows(ndof_mesh) = utils::flatten(utils::reorder_matrix(
			remeshing.displacements(), state.in_node_to_node));
		if (ndof_obstacle > 0)
			sol.bottomRows(ndof_obstacle) = obstacle_sol;

		state.solve_data.rhs_assembler = state.build_rhs_assembler();
		state.init_nonlinear_tensor_solve(sol, time, /*init_time_integrator=*/false);

		if (state.problem->is_time_dependent())
		{
			assert(state.solve_data.time_integrator != nullptr);

			Eigen::MatrixXd projected_quantities = remeshing.projected_quantities();
			assert(projected_quantities.rows() == ndof_mesh);
			assert(projected_quantities.cols() == projection_quantities.cols());
			projected_quantities.conservativeResize(ndof, Eigen::NoChange);
			projected_quantities.bottomRows(ndof_obstacle) = obstacle_projection_quantities;
			// drop the last column (the friction gradient)
			projected_quantities.conservativeResize(Eigen::NoChange, projected_quantities.cols() - 1);

			std::vector<Eigen::VectorXd> x_prevs, v_prevs, a_prevs;
			split_projection_quantities(state, projected_quantities, x_prevs, v_prevs, a_prevs);
			state.solve_data.time_integrator->init(x_prevs, v_prevs, a_prevs, dt);
		}

		// initialize the problem so contact force show up correctly in the output
		state.solve_data.nl_problem->init(sol);
		if (state.solve_data.nl_problem->uses_lagging())
		{
			state.solve_data.friction_form->init_lagging(state.solve_data.time_integrator->x_prev());
			state.solve_data.friction_form->update_lagging(sol);
		}
		state.solve_data.update_barrier_stiffness(sol); // TODO: remove this

		return true;
	}
} // namespace polyfem::mesh
