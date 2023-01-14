#include "Remesh.hpp"

#include <polyfem/mesh/remesh/WildTriRemesher.hpp>
#include <polyfem/mesh/remesh/WildTetRemesher.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <igl/PI.h>
#include <igl/boundary_facets.h>
#include <igl/edges.h>

namespace polyfem::mesh
{
	namespace
	{
		Remesher::BoundaryMap<int> build_boundary_to_id(
			const std::unique_ptr<mesh::Mesh> &mesh,
			const Eigen::VectorXi &in_node_to_node)
		{
			if (mesh->dimension() == 2)
			{
				Remesher::EdgeMap<int> edge_to_boundary_id;
				for (int i = 0; i < mesh->n_edges(); i++)
				{
					const size_t e0 = in_node_to_node[mesh->edge_vertex(i, 0)];
					const size_t e1 = in_node_to_node[mesh->edge_vertex(i, 1)];
					edge_to_boundary_id[{{e0, e1}}] = mesh->get_boundary_id(i);
				}
				return edge_to_boundary_id;
			}
			else
			{
				assert(mesh->dimension() == 3);
				Remesher::FaceMap<int> face_to_boundary_id;
				for (int i = 0; i < mesh->n_faces(); i++)
				{
					std::array<size_t, 3> f =
						{{(size_t)in_node_to_node[mesh->face_vertex(i, 0)],
						  (size_t)in_node_to_node[mesh->face_vertex(i, 1)],
						  (size_t)in_node_to_node[mesh->face_vertex(i, 2)]}};
					face_to_boundary_id[f] = mesh->get_boundary_id(i);
				}
				return face_to_boundary_id;
			}
		}

		void build_edge_energy_maps(
			const State &state,
			const Eigen::MatrixXi &elements,
			const Eigen::MatrixXd &sol,
			Remesher::EdgeMap<double> &elastic_energy,
			Remesher::EdgeMap<double> &contact_energy)
		{
			const size_t n_out_vertices = elements.size();

			Eigen::MatrixXi edges;
			igl::edges(elements, edges);
			Remesher::EdgeMap<std::vector<double>> elastic_multienergy;
			for (const auto &edge : edges.rowwise())
			{
				elastic_multienergy[{{(size_t)edge(0), (size_t)edge(1)}}] = std::vector<double>();
			}

			assert(state.solve_data.elastic_form != nullptr);
			const Eigen::VectorXd elastic_energy_per_element = state.solve_data.elastic_form->value_per_element(sol);
			for (int i = 0; i < elements.rows(); ++i)
			{
				assert(elements.cols() == 3 || elements.cols() == 4);
				const auto &element = elements.row(i);
				const double energy = elastic_energy_per_element[i];

				elastic_multienergy[{{(size_t)element(0), (size_t)element(1)}}].push_back(energy);
				elastic_multienergy[{{(size_t)element(0), (size_t)element(2)}}].push_back(energy);
				elastic_multienergy[{{(size_t)element(1), (size_t)element(2)}}].push_back(energy);
				if (elements.cols() == 4)
				{
					elastic_multienergy[{{(size_t)element(0), (size_t)element(3)}}].push_back(energy);
					elastic_multienergy[{{(size_t)element(1), (size_t)element(3)}}].push_back(energy);
					elastic_multienergy[{{(size_t)element(2), (size_t)element(3)}}].push_back(energy);
				}
			}

			// Average the element energies
			for (const auto &[edge, energies] : elastic_multienergy)
			{
				elastic_energy[edge] = std::accumulate(energies.begin(), energies.end(), 0.0) / energies.size();
			}

			if (state.solve_data.contact_form != nullptr)
			{
				const Eigen::VectorXd contact_energy_per_vertex = state.solve_data.contact_form->value_per_element(sol);
				for (int i = 0; i < edges.rows(); ++i)
				{
					contact_energy[{{(size_t)edges(i, 0), (size_t)edges(i, 1)}}] =
						(contact_energy_per_vertex[edges(i, 0)] + contact_energy_per_vertex[edges(i, 1)]) / 2.0;
				}
			}
		}

		std::shared_ptr<Remesher> create_wild_remeshing(
			State &state,
			const Eigen::VectorXd &obstacle_sol,
			const Eigen::MatrixXd &obstacle_projection_quantities,
			const double time,
			const double current_energy)
		{
			const int dim = state.mesh->dimension();

			std::shared_ptr<Remesher> remeshing;
			if (dim == 2)
				remeshing = std::make_shared<WildTriRemesher>(
					state, utils::unflatten(obstacle_sol, dim), obstacle_projection_quantities,
					time, current_energy);
			else
				remeshing = std::make_shared<WildTetRemesher>(
					state, utils::unflatten(obstacle_sol, dim), obstacle_projection_quantities,
					time, current_energy);

			remeshing->split_tolerance = state.args["space"]["remesh"]["split_tol"];
			remeshing->collapse_tolerance = state.args["space"]["remesh"]["collapse_tol"];
			remeshing->swap_tolerance = state.args["space"]["remesh"]["swap_tol"];
			remeshing->smooth_tolerance = state.args["space"]["remesh"]["smooth_tol"];
			remeshing->n_ring_size = state.args["space"]["remesh"]["n_ring_size"];
			remeshing->flood_fill_rel_area = state.args["space"]["remesh"]["flood_fill_rel_area"];
			remeshing->threshold = state.args["space"]["remesh"]["threshold"];
			remeshing->max_split_depth = state.args["space"]["remesh"]["max_split_depth"];

			return remeshing;
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

		// --------------------------------------------------------------------

		Remesher::EdgeMap<double> elastic_energy, contact_energy;
		build_edge_energy_maps(
			state, elements, sol, elastic_energy, contact_energy);

		// --------------------------------------------------------------------

		Remesher::BoundaryMap<int> boundary_to_id = build_boundary_to_id(state.mesh, state.in_node_to_node);

		const std::vector<int> body_ids = state.mesh->has_body_ids() ? state.mesh->get_body_ids() : std::vector<int>(elements.rows(), 0);
		assert(body_ids.size() == elements.rows());

		// Only remesh the FE mesh
		assert(sol.size() - rest_positions.size() == state.obstacle.n_vertices() * dim);
		const Eigen::MatrixXd mesh_sol = sol.topRows(ndof_mesh);
		const Eigen::MatrixXd obstacle_sol = sol.bottomRows(ndof_obstacle);
		const Eigen::MatrixXd positions = rest_positions + utils::unflatten(mesh_sol, dim);

		// not including current displacement as this will be handled as positions
		Eigen::MatrixXd projection_quantities = Remesher::combine_time_integrator_quantities(
			state.solve_data.time_integrator);
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

		// --------------------------------------------------------------------
		// remesh

		std::shared_ptr<Remesher> remeshing = create_wild_remeshing(
			state, obstacle_sol, obstacle_projection_quantities, time, state.solve_data.nl_problem->value(sol));
		remeshing->init(
			rest_positions, positions, elements, projection_quantities, boundary_to_id, body_ids,
			elastic_energy, contact_energy);

		const bool made_change = remeshing->execute(
			/*split=*/true, /*collapse=*/dim == 2, /*smooth=*/false, /*swap=*/false);

		// remeshing->write_mesh(
		// 	state.resolve_output_path(fmt::format("post_vis_{:03d}.vtu", int(time / dt))));

		if (!made_change)
			return false;

		// --------------------------------------------------------------------
		// create new mesh

		state.mesh = mesh::Mesh::create(remeshing->rest_positions(), remeshing->elements(), /*non_conforming=*/false);

		// set body ids
		state.mesh->set_body_ids(remeshing->body_ids());

		// set boundary ids
		std::vector<int> boundary_ids;
		if (dim == 2)
		{
			const auto remesh_boundary_ids = std::get<Remesher::EdgeMap<int>>(remeshing->boundary_ids());
			boundary_ids = std::vector<int>(state.mesh->n_edges(), -1);
			for (int i = 0; i < state.mesh->n_edges(); i++)
			{
				const size_t e0 = state.mesh->edge_vertex(i, 0);
				const size_t e1 = state.mesh->edge_vertex(i, 1);
				boundary_ids[i] = remesh_boundary_ids.at({{e0, e1}});
			}
		}
		else
		{
			const auto remesh_boundary_ids = std::get<Remesher::FaceMap<int>>(remeshing->boundary_ids());
			boundary_ids = std::vector<int>(state.mesh->n_faces(), -1);
			for (int i = 0; i < state.mesh->n_faces(); i++)
			{
				std::array<size_t, 3> f = {{(size_t)state.mesh->face_vertex(i, 0),
											(size_t)state.mesh->face_vertex(i, 1),
											(size_t)state.mesh->face_vertex(i, 2)}};
				boundary_ids[i] = remesh_boundary_ids.at(f);
			}
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
			remeshing->displacements(), state.in_node_to_node));
		if (ndof_obstacle > 0)
			sol.bottomRows(ndof_obstacle) = obstacle_sol;

		state.solve_data.rhs_assembler = state.build_rhs_assembler();
		state.init_nonlinear_tensor_solve(sol, time, /*init_time_integrator=*/false);

		if (state.problem->is_time_dependent())
		{
			assert(state.solve_data.time_integrator != nullptr);

			Eigen::MatrixXd projected_quantities = remeshing->projection_quantities();
			assert(projected_quantities.rows() == ndof_mesh);
			assert(projected_quantities.cols() == projection_quantities.cols());
			projected_quantities = utils::reorder_matrix(
				projected_quantities, state.in_node_to_node, /*out_blocks=*/-1, dim);
			projected_quantities.conservativeResize(ndof, Eigen::NoChange);
			projected_quantities.bottomRows(ndof_obstacle) = obstacle_projection_quantities;
			// drop the last column (the friction gradient)
			projected_quantities.conservativeResize(Eigen::NoChange, projected_quantities.cols() - 1);

			std::vector<Eigen::VectorXd> x_prevs, v_prevs, a_prevs;
			Remesher::split_time_integrator_quantities(
				projected_quantities, dim, x_prevs, v_prevs, a_prevs);
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
