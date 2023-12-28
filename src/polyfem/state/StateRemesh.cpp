#include <polyfem/State.hpp>

#include <polyfem/mesh/remesh/PhysicsRemesher.hpp>
#include <polyfem/mesh/remesh/SizingFieldRemesher.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

#include <igl/edges.h>

namespace polyfem
{
	using namespace mesh;

	namespace
	{
		/// @brief Build a map from boundary facets to boundary ids.
		/// @param mesh The mesh.
		/// @param in_node_to_node The map from the input node indices to current node indices.
		/// @return The map from boundary facets to boundary ids.
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

#ifndef NDEBUG
					if (mesh->is_boundary_face(i))
						assert(face_to_boundary_id[f] >= 0);
					else
						assert(face_to_boundary_id[f] == -1);
#endif
				}
				return face_to_boundary_id;
			}
		}

		/// @brief Build a map from edges to the elastic and contact energies.
		/// @param[in] state The state.
		/// @param[in] vertices Vertices of the mesh.
		/// @param[in] elements Elements of the mesh (P1 triangles or tetrahedra).
		/// @param[in] sol The current solution.
		/// @param[out] elastic_energy The map from edges to elastic energy.
		/// @param[out] contact_energy The map from edges to contact energy.
		void build_edge_energy_maps(
			const State &state,
			const Eigen::MatrixXd &vertices,
			const Eigen::MatrixXi &elements,
			const Eigen::MatrixXd &sol,
			Remesher::EdgeMap<double> &elastic_energy,
			Remesher::EdgeMap<double> &contact_energy)
		{
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
				elastic_energy[edge] =
					std::reduce(energies.begin(), energies.end()) / energies.size();
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

			const std::string type = state.args["space"]["remesh"]["type"];
			std::shared_ptr<Remesher> remeshing = nullptr;
			if (type == "physics")
			{
				if (dim == 2)
					remeshing = std::make_shared<PhysicsTriRemesher>(
						state, utils::unflatten(obstacle_sol, dim), obstacle_projection_quantities,
						time, current_energy);
				else
					remeshing = std::make_shared<PhysicsTetRemesher>(
						state, utils::unflatten(obstacle_sol, dim), obstacle_projection_quantities,
						time, current_energy);
			}
			else if (type == "sizing_field")
			{
				if (dim == 2)
					remeshing = std::make_shared<SizingFieldTriRemesher>(
						state, utils::unflatten(obstacle_sol, dim), obstacle_projection_quantities,
						time, current_energy);
				else
					remeshing = std::make_shared<SizingFieldTetRemesher>(
						state, utils::unflatten(obstacle_sol, dim), obstacle_projection_quantities,
						time, current_energy);
			}
			assert(remeshing != nullptr);
			return remeshing;
		}
	} // namespace

	bool State::remesh(const double time, const double dt, Eigen::MatrixXd &sol)
	{
		const int dim = mesh->dimension();
		int ndof = sol.size();
		assert(sol.cols() == 1);
		int ndof_mesh = mesh->n_vertices() * dim;
		int ndof_obstacle = obstacle.n_vertices() * dim;
		assert(ndof == ndof_mesh + ndof_obstacle);

		Eigen::MatrixXd rest_positions;
		Eigen::MatrixXi elements;
		build_mesh_matrices(rest_positions, elements);
		assert(rest_positions.size() == ndof_mesh);

		// --------------------------------------------------------------------

		Remesher::EdgeMap<double> elastic_energy, contact_energy;
		build_edge_energy_maps(
			*this, rest_positions, elements, sol, elastic_energy, contact_energy);

		// --------------------------------------------------------------------

		const Remesher::BoundaryMap<int> boundary_to_id = build_boundary_to_id(mesh, in_node_to_node);

		const std::vector<int> body_ids = mesh->has_body_ids() ? mesh->get_body_ids() : std::vector<int>(elements.rows(), 0);
		assert(body_ids.size() == elements.rows());

		// Only remesh the FE mesh
		assert(sol.size() - rest_positions.size() == obstacle.n_vertices() * dim);
		const Eigen::MatrixXd mesh_sol = sol.topRows(ndof_mesh);
		const Eigen::MatrixXd obstacle_sol = sol.bottomRows(ndof_obstacle);
		const Eigen::MatrixXd positions = rest_positions + utils::unflatten(mesh_sol, dim);

		// not including current displacement as this will be handled as positions
		Eigen::MatrixXd projection_quantities = Remesher::combine_time_integrator_quantities(
			solve_data.time_integrator);
		assert(projection_quantities.rows() == ndof);

		const Eigen::MatrixXd obstacle_projection_quantities = projection_quantities.bottomRows(ndof_obstacle);
		projection_quantities.conservativeResize(ndof_mesh, Eigen::NoChange);

		// --------------------------------------------------------------------
		// remesh

		std::shared_ptr<Remesher> remeshing = create_wild_remeshing(
			*this, obstacle_sol, obstacle_projection_quantities, time, solve_data.nl_problem->value(sol));
		remeshing->init(
			rest_positions, positions, elements, projection_quantities, boundary_to_id, body_ids,
			elastic_energy, contact_energy);

		const bool made_change = remeshing->execute();

		if (!made_change)
			return false;

		// --------------------------------------------------------------------
		// create new mesh

		mesh = mesh::Mesh::create(remeshing->rest_positions(), remeshing->elements(), /*non_conforming=*/false);

		// set body ids
		mesh->set_body_ids(remeshing->body_ids());

		// set boundary ids
		std::vector<int> boundary_ids;
		if (dim == 2)
		{
			const auto remesh_boundary_ids = std::get<Remesher::EdgeMap<int>>(remeshing->boundary_ids());
			boundary_ids = std::vector<int>(mesh->n_edges(), -1);
			for (int i = 0; i < mesh->n_edges(); i++)
			{
				const size_t e0 = mesh->edge_vertex(i, 0);
				const size_t e1 = mesh->edge_vertex(i, 1);
				boundary_ids[i] = remesh_boundary_ids.at({{e0, e1}});
			}
		}
		else
		{
			const auto remesh_boundary_ids = std::get<Remesher::FaceMap<int>>(remeshing->boundary_ids());
			boundary_ids = std::vector<int>(mesh->n_faces(), -1);
			for (int i = 0; i < mesh->n_faces(); i++)
			{
				const std::array<size_t, 3> f = {{(size_t)mesh->face_vertex(i, 0),
												  (size_t)mesh->face_vertex(i, 1),
												  (size_t)mesh->face_vertex(i, 2)}};
				boundary_ids[i] = remesh_boundary_ids.at(f);
			}
		}
		mesh->set_boundary_ids(boundary_ids);

		// load mesh (and set materials) (will also reload obstacles from disk)
		load_mesh();

		// --------------------------------------------------------------------

		build_basis();
		assemble_rhs();
		assemble_mass_mat();

		// --------------------------------------------------------------------

		const int old_ndof = ndof;
		const int old_ndof_mesh = ndof_mesh;
		const int old_ndof_obstacle = ndof_obstacle;

		ndof_mesh = mesh->n_vertices() * dim;
		ndof_obstacle = obstacle.n_vertices() * dim;
		assert(ndof_obstacle == old_ndof_obstacle);
		ndof = n_bases * dim;
		assert(ndof == ndof_mesh + ndof_obstacle);

		sol.resize(ndof, 1);
		sol.topRows(ndof_mesh) = utils::flatten(utils::reorder_matrix(remeshing->displacements(), in_node_to_node));
		if (ndof_obstacle > 0)
			sol.bottomRows(ndof_obstacle) = obstacle_sol;

		solve_data.rhs_assembler = build_rhs_assembler();
		if (problem->is_time_dependent())
		{
			assert(solve_data.time_integrator != nullptr);

			Eigen::MatrixXd projected_quantities = remeshing->projection_quantities();
			assert(projected_quantities.rows() == ndof_mesh);
			assert(projected_quantities.cols() == projection_quantities.cols());
			projected_quantities = utils::reorder_matrix(
				projected_quantities, in_node_to_node, /*out_blocks=*/-1, dim);
			projected_quantities.conservativeResize(ndof, Eigen::NoChange);
			projected_quantities.bottomRows(ndof_obstacle) = obstacle_projection_quantities;

			Eigen::MatrixXd x_prevs, v_prevs, a_prevs;
			Remesher::split_time_integrator_quantities(
				projected_quantities, dim, x_prevs, v_prevs, a_prevs);
			solve_data.time_integrator->init(x_prevs, v_prevs, a_prevs, dt);
		}
		init_nonlinear_tensor_solve(sol, time, /*init_time_integrator=*/false);

		// initialize the problem so contact force show up correctly in the output
		solve_data.nl_problem->update_quantities(time, solve_data.time_integrator->x_prev());
		solve_data.nl_problem->init(sol);
		solve_data.nl_problem->init_lagging(solve_data.time_integrator->x_prev());
		solve_data.nl_problem->update_lagging(sol, /*iter_num=*/0);

		return true;
	}
} // namespace polyfem