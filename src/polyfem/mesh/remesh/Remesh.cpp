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
		bool points_equal(const RowVectorNd &p0, const RowVectorNd &p1)
		{
			return (p0 - p1).norm() < 1e-15;
		}

		bool edges_equal(const RowVectorNd &ea0, const RowVectorNd &ea1, const RowVectorNd &eb0, const RowVectorNd &eb1)
		{
			return (points_equal(ea0, eb0) && points_equal(ea1, eb1)) || (points_equal(ea0, eb1) && points_equal(ea1, eb0));
		}
	} // namespace

	void remesh(State &state, const double time, const double dt)
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

		assert(!state.mesh->is_volume());
		WildRemeshing2D remeshing(state.obstacle);
		remeshing.create_mesh(
			rest_positions,
			rest_positions + utils::unflatten(state.sol, dim),
			utils::unflatten(state.solve_data.time_integrator->v_prev(), dim),
			utils::unflatten(state.solve_data.time_integrator->a_prev(), dim),
			elements,
			edge_to_boundary_id,
			body_ids);

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

		// std::vector<int> mesh_edge_to_remesh_edge(state.mesh->n_edges(), -1);
		// const Eigen::MatrixXi edges = remeshing.edges();
		// for (int i = 0; i < state.mesh->n_edges(); i++)
		// {
		// 	int e0 = state.mesh->edge_vertex(i, 0);
		// 	int e1 = state.mesh->edge_vertex(i, 1);
		// 	for (int j = 0; j < edges.rows(); j++)
		// 	{
		// 		// TODO: find a better way to match edges
		// 		if (edges_equal(rest_positions.row(e0), rest_positions.row(e1), remeshing.rest_positions().row(edges(j, 0)), remeshing.rest_positions().row(edges(j, 1))))
		// 		{
		// 			mesh_edge_to_remesh_edge[i] = j;
		// 			break;
		// 		}
		// 	}
		// 	assert(mesh_edge_to_remesh_edge[i] >= 0);
		// }

		// const std::vector<int> remesh_boundary_ids = remeshing.boundary_ids();
		const WildRemeshing2D::EdgeMap remesh_boundary_ids = remeshing.boundary_ids();
		std::vector<int> boundary_ids(state.mesh->n_edges(), -1);
		for (int i = 0; i < state.mesh->n_edges(); i++)
		{
			// boundary_ids[i] = remesh_boundary_ids[mesh_edge_to_remesh_edge[i]];
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

		const Eigen::MatrixXd U = remeshing.displacements();
		const Eigen::MatrixXd V = remeshing.velocities();
		const Eigen::MatrixXd A = remeshing.accelerations();
		Eigen::MatrixXd U_reordered, V_reordered, A_reordered;
		U_reordered.resizeLike(U);
		V_reordered.resizeLike(V);
		A_reordered.resizeLike(A);
		assert(state.in_node_to_node.size() == state.mesh->n_vertices());
		for (int i = 0; i < state.mesh->n_vertices(); ++i)
		{
			U_reordered.row(state.in_node_to_node[i]) = U.row(i);
			V_reordered.row(state.in_node_to_node[i]) = V.row(i);
			A_reordered.row(state.in_node_to_node[i]) = A.row(i);
		}
		const Eigen::VectorXd displacements = utils::flatten(U_reordered);
		const Eigen::VectorXd velocities = utils::flatten(V_reordered);
		const Eigen::VectorXd accelerations = utils::flatten(A_reordered);

		// x = min Inertia(x^t)
		// x = remesh(x)
		// x = min (Inertia(x) = 0)
		// loop

		state.sol = displacements;

		state.solve_data.rhs_assembler = state.build_rhs_assembler();
		state.init_nonlinear_tensor_solve(time);
		if (state.problem->is_time_dependent())
		{
			state.solve_data.time_integrator->init(
				displacements, velocities, accelerations, dt);
		}
	}
} // namespace polyfem::mesh
