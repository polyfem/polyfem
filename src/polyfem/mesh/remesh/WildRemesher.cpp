#include "WildRemesher.hpp"

#include <polyfem/mesh/remesh/wild_remesh/LocalMesh.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

#include <wmtk/utils/TupleUtils.hpp>

#include <unordered_map>

#define VERTEX_ATTRIBUTE_GETTER(name, attribute)                                                     \
	template <class WMTKMesh>                                                                        \
	Eigen::MatrixXd WildRemesher<WMTKMesh>::name() const                                             \
	{                                                                                                \
		Eigen::MatrixXd attributes = Eigen::MatrixXd::Constant(WMTKMesh::vert_capacity(), DIM, NaN); \
		for (const Tuple &t : WMTKMesh::get_vertices())                                              \
			attributes.row(t.vid(*this)) = vertex_attrs[t.vid(*this)].attribute;                     \
		return attributes;                                                                           \
	}

#define VERTEX_ATTRIBUTE_SETTER(name, attribute)                                 \
	template <class WMTKMesh>                                                    \
	void WildRemesher<WMTKMesh>::name(const Eigen::MatrixXd &attributes)         \
	{                                                                            \
		for (const Tuple &t : WMTKMesh::get_vertices())                          \
			vertex_attrs[t.vid(*this)].attribute = attributes.row(t.vid(*this)); \
	}

namespace polyfem::mesh
{
	static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

	template <class WMTKMesh>
	WildRemesher<WMTKMesh>::WildRemesher(
		const State &state,
		const Eigen::MatrixXd &obstacle_displacements,
		const Eigen::MatrixXd &obstacle_vals,
		const double current_time,
		const double starting_energy)
		: Remesher(state, obstacle_displacements, obstacle_vals, current_time, starting_energy),
		  WMTKMesh()
	{
	}

	template <typename WMTKMesh>
	Remesher::EdgeMap<typename WildRemesher<WMTKMesh>::EdgeAttributes::EnergyRank>
	rank_edges(const Remesher::EdgeMap<double> &edge_energy, const double threshold)
	{
		if (edge_energy.empty())
			return Remesher::EdgeMap<typename WildRemesher<WMTKMesh>::EdgeAttributes::EnergyRank>();

		double min_energy = std::numeric_limits<double>::infinity();
		double max_energy = -std::numeric_limits<double>::infinity();
		std::vector<double> sorted_energies;
		sorted_energies.reserve(edge_energy.size());
		for (const auto &[edge, energy] : edge_energy)
		{
			min_energy = std::min(min_energy, energy);
			max_energy = std::max(max_energy, energy);
			sorted_energies.push_back(energy);
		}
		std::sort(sorted_energies.begin(), sorted_energies.end());

		assert(0.0 <= threshold && threshold <= 1.0);

		const double top_energy_threshold = (max_energy - min_energy) * threshold + min_energy;
		const double bottom_energy_threshold = (max_energy - min_energy) * (1 - threshold) + min_energy;

		const double top_element_threshold = sorted_energies[int(sorted_energies.size() * threshold)];
		const double bottom_element_threshold = sorted_energies[int(sorted_energies.size() * (1 - threshold))];

		const double top_threshold = std::max(std::min(top_energy_threshold, top_element_threshold), 1e-12);
		const double bottom_threshold = bottom_energy_threshold;
		assert(bottom_threshold < top_threshold);

		logger().info("min energy: {}, max energy: {}, thresholds: {}, {}", min_energy, max_energy, bottom_threshold, top_threshold);

		Remesher::EdgeMap<typename WildRemesher<WMTKMesh>::EdgeAttributes::EnergyRank> edge_ranks;
		for (const auto &[edge, energy] : edge_energy)
		{
			if (energy >= top_threshold)
				edge_ranks[edge] = WildRemesher<WMTKMesh>::EdgeAttributes::EnergyRank::TOP;
			else if (energy <= bottom_threshold)
				edge_ranks[edge] = WildRemesher<WMTKMesh>::EdgeAttributes::EnergyRank::BOTTOM;
			else
				edge_ranks[edge] = WildRemesher<WMTKMesh>::EdgeAttributes::EnergyRank::MIDDLE;
		}

		return edge_ranks;
	}

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::init(
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXi &elements,
		const Eigen::MatrixXd &projection_quantities,
		const BoundaryMap<int> &boundary_to_id,
		const std::vector<int> &body_ids,
		const EdgeMap<double> &elastic_energy,
		const EdgeMap<double> &contact_energy)
	{
		Remesher::init(
			rest_positions, positions, elements, projection_quantities,
			boundary_to_id, body_ids, elastic_energy, contact_energy);

		total_volume = 0;
		for (const Tuple &t : get_elements())
			total_volume += element_volume(t);
		assert(total_volume > 0);

#ifndef NDEBUG
		assert(get_elements().size() == elements.rows());
		for (const Tuple &t : get_elements())
			assert(!is_inverted(t));
#endif

		const auto edge_elastic_ranks = rank_edges<WMTKMesh>(elastic_energy, threshold);
		const auto edge_contact_ranks = rank_edges<WMTKMesh>(contact_energy, threshold);

		EdgeMap<typename EdgeAttributes::EnergyRank> edge_ranks;
		for (const auto &[edge, elastic_rank] : edge_elastic_ranks)
		{
			const auto contact_rank = edge_contact_ranks.empty() ? elastic_rank : edge_contact_ranks.at(edge);
			if (elastic_rank == EdgeAttributes::EnergyRank::TOP || contact_rank == EdgeAttributes::EnergyRank::TOP)
				edge_ranks[edge] = EdgeAttributes::EnergyRank::TOP;
			else if (elastic_rank == EdgeAttributes::EnergyRank::BOTTOM && contact_rank == EdgeAttributes::EnergyRank::BOTTOM)
				edge_ranks[edge] = EdgeAttributes::EnergyRank::BOTTOM;
			else
				edge_ranks[edge] = EdgeAttributes::EnergyRank::MIDDLE;
		}

		for (const Tuple &edge : WMTKMesh::get_edges())
		{
			const size_t e0 = edge.vid(*this);
			const size_t e1 = edge.switch_vertex(*this).vid(*this);
			edge_attr(edge.eid(*this)).energy_rank = edge_ranks.at({{e0, e1}});
		}

		write_edge_ranks_mesh(edge_elastic_ranks, edge_contact_ranks);
	}

	// -------------------------------------------------------------------------
	// Getters

	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	// 2D (Triangle mesh)

	template <>
	void WildRemesher<wmtk::TriMesh>::set_boundary_ids(const BoundaryMap<int> &boundary_to_id)
	{
		assert(std::holds_alternative<EdgeMap<int>>(boundary_to_id));
		const EdgeMap<int> &edge_to_boundary_id = std::get<EdgeMap<int>>(boundary_to_id);
		for (const Tuple &edge : get_edges())
		{
			const size_t e0 = edge.vid(*this);
			const size_t e1 = edge.switch_vertex(*this).vid(*this);
			boundary_attrs[edge.eid(*this)].boundary_id = edge_to_boundary_id.at({{e0, e1}});
		}
	}

	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	// 3D (Tetrahedron mesh)

	template <>
	void WildRemesher<wmtk::TetMesh>::set_boundary_ids(const BoundaryMap<int> &boundary_to_id)
	{
		assert(std::holds_alternative<FaceMap<int>>(boundary_to_id));
		const FaceMap<int> &face_to_boundary_id = std::get<FaceMap<int>>(boundary_to_id);
		for (const Tuple &face : get_faces())
		{
			const size_t f0 = face.vid(*this);
			const size_t f1 = face.switch_vertex(*this).vid(*this);
			const size_t f2 = face.switch_edge(*this).switch_vertex(*this).vid(*this);

			boundary_attrs[face.fid(*this)].boundary_id = face_to_boundary_id.at({{f0, f1, f2}});
		}
	}

	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	// ND

	VERTEX_ATTRIBUTE_GETTER(rest_positions, rest_position)
	VERTEX_ATTRIBUTE_GETTER(positions, position)
	VERTEX_ATTRIBUTE_GETTER(displacements, displacement())

	template <class WMTKMesh>
	Eigen::MatrixXi WildRemesher<WMTKMesh>::edges() const
	{
		const std::vector<Tuple> edges = WMTKMesh::get_edges();
		Eigen::MatrixXi E = Eigen::MatrixXi::Constant(edges.size(), 2, -1);
		for (int i = 0; i < edges.size(); ++i)
		{
			const Tuple &e = edges[i];
			E(i, 0) = e.vid(*this);
			E(i, 1) = e.switch_vertex(*this).vid(*this);
		}
		return E;
	}

	template <class WMTKMesh>
	Eigen::MatrixXi WildRemesher<WMTKMesh>::elements() const
	{
		const std::vector<Tuple> elements = get_elements();
		Eigen::MatrixXi F = Eigen::MatrixXi::Constant(elements.size(), dim() + 1, -1);
		for (size_t i = 0; i < elements.size(); i++)
		{
			const auto vids = element_vids(elements[i]);

			for (int j = 0; j < vids.size(); j++)
			{
				F(i, j) = vids[j];
			}
		}
		return F;
	}

	template <class WMTKMesh>
	Eigen::MatrixXd WildRemesher<WMTKMesh>::projection_quantities() const
	{
		Eigen::MatrixXd projection_quantities =
			Eigen::MatrixXd::Constant(dim() * WMTKMesh::vert_capacity(), n_quantities(), NaN);

		for (const Tuple &t : WMTKMesh::get_vertices())
		{
			const int vi = t.vid(*this);
			projection_quantities.middleRows(dim() * vi, dim()) = vertex_attrs[vi].projection_quantities;
		}

		return projection_quantities;
	}

	template <>
	WildRemesher<wmtk::TriMesh>::BoundaryMap<int> WildRemesher<wmtk::TriMesh>::boundary_ids() const
	{
		const std::vector<Tuple> edges = get_edges();
		EdgeMap<int> boundary_ids;
		for (const Tuple &edge : edges)
		{
			const size_t e0 = edge.vid(*this);
			const size_t e1 = edge.switch_vertex(*this).vid(*this);
			boundary_ids[{{e0, e1}}] = boundary_attrs[edge.eid(*this)].boundary_id;
		}
		return boundary_ids;
	}

	template <>
	WildRemesher<wmtk::TetMesh>::BoundaryMap<int> WildRemesher<wmtk::TetMesh>::boundary_ids() const
	{
		const std::vector<Tuple> faces = get_faces();
		FaceMap<int> boundary_ids;
		for (const Tuple &face : faces)
		{
			const size_t f0 = face.vid(*this);
			const size_t f1 = face.switch_vertex(*this).vid(*this);
			const size_t f2 = face.switch_edge(*this).switch_vertex(*this).vid(*this);

			boundary_ids[{{f0, f1, f2}}] = boundary_attrs[face.fid(*this)].boundary_id;
		}
		return boundary_ids;
	}

	template <class WMTKMesh>
	std::vector<int> WildRemesher<WMTKMesh>::body_ids() const
	{
		const std::vector<Tuple> elements = get_elements();
		std::vector<int> body_ids(elements.size(), -1);
		for (size_t i = 0; i < elements.size(); i++)
		{
			body_ids[i] = element_attrs[element_id(elements[i])].body_id;
		}
		return body_ids;
	}

	template <class WMTKMesh>
	std::vector<int> WildRemesher<WMTKMesh>::boundary_nodes(
		const Eigen::VectorXi &vertex_to_basis) const
	{
		std::vector<int> boundary_nodes;

		// TODO: get this from state rather than building it
		std::unordered_map<int, std::array<bool, DIM>> bc_ids;
		{
			assert(state.args["boundary_conditions"]["dirichlet_boundary"].is_array());
			const std::vector<json> bcs = state.args["boundary_conditions"]["dirichlet_boundary"];
			for (const json &bc : bcs)
			{
				assert(bc["dimension"].size() == dim());
				bc_ids[bc["id"].get<int>()] = bc["dimension"];
			}
		}

		for (const Tuple &t : boundary_facets())
		{
			const auto bc = bc_ids.find(boundary_attrs[facet_id(t)].boundary_id);

			if (bc == bc_ids.end())
				continue;

			for (const size_t vid : boundary_facet_vids(t))
				for (int d = 0; d < dim(); ++d)
					if (bc->second[d])
						boundary_nodes.push_back(dim() * vertex_to_basis[vid] + d);
		}

		// Sort and remove the duplicate boundary_nodes.
		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto new_end = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.erase(new_end, boundary_nodes.end());

		return boundary_nodes;
	}

	// -------------------------------------------------------------------------
	// Setters

	VERTEX_ATTRIBUTE_SETTER(set_rest_positions, rest_position)
	VERTEX_ATTRIBUTE_SETTER(set_positions, position)

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::set_projection_quantities(
		const Eigen::MatrixXd &projection_quantities)
	{
		assert(projection_quantities.rows() == dim() * WMTKMesh::vert_capacity());
		m_n_quantities = projection_quantities.cols();

		for (const Tuple &t : WMTKMesh::get_vertices())
		{
			const int vi = t.vid(*this);
			vertex_attrs[vi].projection_quantities =
				projection_quantities.middleRows(dim() * vi, dim());
		}
	}

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::set_fixed(
		const std::vector<bool> &fixed)
	{
		assert(fixed.size() == WMTKMesh::vert_capacity());
		for (const Tuple &t : WMTKMesh::get_vertices())
			vertex_attrs[t.vid(*this)].fixed = fixed[t.vid(*this)];
	}

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::set_body_ids(
		const std::vector<int> &body_ids)
	{
		const std::vector<Tuple> elements = get_elements();
		for (int i = 0; i < elements.size(); ++i)
		{
			element_attrs[element_id(elements[i])].body_id = body_ids.at(i);
		}
	}

	// -------------------------------------------------------------------------

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::invariants(const std::vector<Tuple> &new_tris)
	{
		// for (auto &t : new_tris)
		for (auto &t : get_elements())
		{
			if (is_inverted(t))
			{
				static int inversion_cnt = 0;
				write_mesh(state.resolve_output_path(fmt::format("inversion_{:04d}.vtu", inversion_cnt++)));
				log_and_throw_error("Inverted element found, invariants violated");
				return false;
			}
		}
		return true;
	}

	// -------------------------------------------------------------------------
	// Utils

	template <class WMTKMesh>
	double WildRemesher<WMTKMesh>::edge_length(const Tuple &e) const
	{
		const auto &e0 = vertex_attrs[e.vid(*this)].position;
		const auto &e1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].position;
		return (e1 - e0).norm();
	}

	template <class WMTKMesh>
	Eigen::VectorXd WildRemesher<WMTKMesh>::edge_adjacent_element_volumes(const Tuple &e) const
	{
		double vol_tol;
		std::vector<Tuple> adjacent_elements;
		if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
		{
			adjacent_elements = {{e}};
			const std::optional<Tuple> f = e.switch_face(*this);
			if (f.has_value())
				adjacent_elements.push_back(f.value());
		}
		else
		{
			adjacent_elements = get_incident_elements_for_edge(e);
		}

		Eigen::VectorXd adjacent_element_volumes(adjacent_elements.size());
		for (int i = 0; i < adjacent_elements.size(); ++i)
			adjacent_element_volumes[i] = element_volume(adjacent_elements[i]);

		return adjacent_element_volumes;
	}

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::extend_local_patch(std::vector<Tuple> &patch) const
	{
		const size_t starting_size = patch.size();

		std::unordered_set<size_t> element_ids;
		for (const Tuple &t : patch)
			element_ids.insert(element_id(t));

		for (size_t i = 0; i < starting_size; ++i)
		{
			const size_t id = element_id(patch[i]);
			for (int j = 0; j < EDGES_PER_ELEMENT; ++j)
			{
				const Tuple e = WMTKMesh::tuple_from_edge(id, j);

				for (const Tuple &t : get_incident_elements_for_edge(e))
				{
					const size_t t_id = element_id(t);
					if (element_ids.find(t_id) == element_ids.end())
					{
						patch.push_back(t);
						element_ids.insert(t_id);
					}
				}
			}
		}
	}

	template <class WMTKMesh>
	std::vector<typename WMTKMesh::Tuple>
	WildRemesher<WMTKMesh>::local_mesh_tuples(const Eigen::Matrix<double, DIM, 1> &center) const
	{
		return LocalMesh<WildRemesher<WMTKMesh>>::ball_selection(
			*this, center, flood_fill_rel_area * total_volume);
	}

	template <class WMTKMesh>
	double WildRemesher<WMTKMesh>::local_mesh_energy(
		const Eigen::Matrix<double, DIM, 1> &local_mesh_center) const
	{
		using namespace polyfem::solver;

		const std::vector<Tuple> local_mesh_tuples = this->local_mesh_tuples(local_mesh_center);

		const bool include_global_boundary =
			state.args["contact"]["enabled"].get<bool>()
			&& std::any_of(local_mesh_tuples.begin(), local_mesh_tuples.end(), [&](const Tuple &t) {
				   const size_t tid = element_id(t);
				   for (int i = 0; i < FACETS_PER_ELEMENT; ++i)
					   if (is_on_boundary(tuple_from_facet(tid, i)))
						   return true;
				   return false;
			   });

		LocalMesh local_mesh(*this, local_mesh_tuples, include_global_boundary);

		const std::vector<polyfem::basis::ElementBases> bases = local_bases(local_mesh);
		const std::vector<int> boundary_nodes = local_boundary_nodes(local_mesh);
		assembler::AssemblerUtils &assembler = init_assembler(local_mesh.body_ids());
		SolveData solve_data;
		assembler::AssemblyValsCache ass_vals_cache;
		Eigen::SparseMatrix<double> mass;
		ipc::CollisionMesh collision_mesh;

		local_solve_data(
			local_mesh, bases, boundary_nodes, assembler, include_global_boundary,
			solve_data, ass_vals_cache, mass, collision_mesh);

		const Eigen::MatrixXd sol = utils::flatten(local_mesh.displacements());

		return solve_data.nl_problem->value(sol);
	}

	template <class WMTKMesh>
	typename WildRemesher<WMTKMesh>::Operations
	WildRemesher<WMTKMesh>::renew_neighbor_tuples(
		const std::string &op,
		const std::vector<Tuple> &elements) const
	{
		// POLYFEM_REMESHER_SCOPED_TIMER("Renew neighbor tuples");
		assert(elements.size() == 1);
		assert(op != "vertex_smooth");

		Eigen::Matrix<double, DIM, 1> center;
		if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
		{
			if (op == "edge_split")
			{
				center = vertex_attrs[elements[0].switch_vertex(*this).vid(*this)].rest_position;
			}
			else if (op == "edge_swap")
			{
				center = (vertex_attrs[elements[0].vid(*this)].rest_position
						  + vertex_attrs[elements[0].switch_vertex(*this).vid(*this)].rest_position)
						 / 2.0;
			}
			else // if (op == "edge_collapse" || op == "vertex_smooth")
			{
				center = vertex_attrs[elements[0].vid(*this)].rest_position;
			}
		}
		else
		{
			assert(op == "edge_split" || op == "edge_collapse");
			center = vertex_attrs[elements[0].vid(*this)].rest_position;
		}

		// return all edges affected by local relaxation
		std::vector<Tuple> local_mesh_tuples = this->local_mesh_tuples(center);
		extend_local_patch(local_mesh_tuples);

		std::vector<Tuple> edges;
		for (const Tuple &t : local_mesh_tuples)
		{
			const size_t t_id = element_id(t);

			for (auto j = 0; j < EDGES_PER_ELEMENT; j++)
			{
				const Tuple e = WMTKMesh::tuple_from_edge(t_id, j);
				const size_t e_id = e.eid(*this);

				if (op == "edge_split" && edge_attr(e_id).energy_rank != EdgeAttributes::EnergyRank::TOP)
					continue;
				else if (op == "edge_collapse" && edge_attr(e_id).energy_rank != EdgeAttributes::EnergyRank::BOTTOM)
					continue;

				edges.push_back(e);
			}
		}
		wmtk::unique_edge_tuples(*this, edges);

		Operations new_ops;
		for (auto &e : edges)
			new_ops.emplace_back(op, e);

		return new_ops;
	}

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::write_priority_queue_mesh(const std::string &path, const Tuple &e) const
	{
		constexpr double tol = 1e-14; // tolerance allowed in recomputed values

		// Save the edge energy and its position in the priority queue
		std::unordered_map<size_t, std::tuple<double, double, int>> edge_to_fields;

		// The current tuple was popped from the queue, so we need to recompute its energy
		const double current_edge_energy = edge_elastic_energy(e);
		edge_to_fields[e.eid(*this)] = std::make_tuple(current_edge_energy, 0, 0);

		// NOTE: this is not thread-safe
		auto queue = executor.serial_queue();

		// Also check that the energy is consistent with the priority queue values
		bool energies_match = true;

		for (int i = 1; !queue.empty(); ++i)
		{
			std::tuple<double, std::string, Tuple, size_t> tmp;
			bool pop_success = queue.try_pop(tmp);
			assert(pop_success);
			const auto &[energy, op, t, _] = tmp;

			// Some tuple in the queue might not be valid anymore
			if (!t.is_valid(*this))
			{
				--i; // don't count this tuple
				continue;
			}

			// assert(t.eid(*this) != e.eid(*this)); // this should have been popped

			// Check that the energy is consistent with the priority queue values
			const double recomputed_energy = edge_elastic_energy(t);
			const double diff = energy - recomputed_energy;
			if (abs(diff) >= tol)
			{
				logger().error(
					"Energy mismatch: {} vs {}; diff={:g}",
					energy, recomputed_energy, diff);
				energies_match = false;
			}

			// Check that the current edge has the highes priority
			assert(current_edge_energy - energy >= -tol); // account for numerical error

			// Save the edge energy and its position in the priority queue
			edge_to_fields[t.eid(*this)] = std::make_tuple(energy, abs(diff), i);
		}
		assert(energies_match);

		const std::vector<Tuple> edges = WMTKMesh::get_edges();

		// Create two vertices per edge to get per edge values.
		const int n_vertices = 2 * edges.size();

		std::vector<std::vector<int>> elements(edges.size(), std::vector<int>(2));
		Eigen::MatrixXd rest_positions(n_vertices, dim());
		Eigen::MatrixXd displacements(n_vertices, dim());
		Eigen::VectorXd edge_energies(n_vertices);
		Eigen::VectorXd edge_energy_diffs(n_vertices);
		Eigen::VectorXd edge_orders(n_vertices);

		for (int ei = 0; ei < edges.size(); ei++)
		{
			const std::array<size_t, 2> vids = {{
				edges[ei].vid(*this),
				edges[ei].switch_vertex(*this).vid(*this),
			}};

			double edge_energy, edge_energy_diff, edge_order;
			const auto &itr = edge_to_fields.find(edges[ei].eid(*this));
			if (itr != edge_to_fields.end())
				std::tie(edge_energy, edge_energy_diff, edge_order) = itr->second;
			else
				edge_energy = edge_energy_diff = edge_order = NaN;

			for (int vi = 0; vi < vids.size(); ++vi)
			{
				elements[ei][vi] = 2 * ei + vi;
				rest_positions.row(elements[ei][vi]) = vertex_attrs[vids[vi]].rest_position;
				displacements.row(elements[ei][vi]) = vertex_attrs[vids[vi]].displacement();
				edge_energies(elements[ei][vi]) = edge_energy;
				edge_energy_diffs(elements[ei][vi]) = edge_energy_diff;
				edge_orders(elements[ei][vi]) = edge_order;
			}
		}

		io::VTUWriter writer;
		writer.add_field("displacement", displacements);
		writer.add_field("edge_energy", edge_energies);
		writer.add_field("edge_energy_diff", edge_energy_diffs);
		writer.add_field("operation_order", edge_orders);
		writer.write_mesh(path, rest_positions, elements, /*is_simplicial=*/true);
	}

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::write_edge_ranks_mesh(
		const EdgeMap<typename EdgeAttributes::EnergyRank> &elastic_ranks,
		const EdgeMap<typename EdgeAttributes::EnergyRank> &contact_ranks) const
	{
		const std::vector<Tuple> edges = WMTKMesh::get_edges();

		// Create two vertices per edge to get per edge values.
		const int n_vertices = 2 * edges.size();

		std::vector<std::vector<int>> elements(edges.size(), std::vector<int>(2));
		Eigen::MatrixXd rest_positions(n_vertices, dim());
		Eigen::MatrixXd displacements(n_vertices, dim());
		Eigen::VectorXd energy_ranks(n_vertices);
		Eigen::VectorXd elastic_energy_ranks(n_vertices);
		Eigen::VectorXd contact_energy_ranks(n_vertices);

		for (int ei = 0; ei < edges.size(); ei++)
		{
			const std::array<size_t, 2> vids = {{
				edges[ei].vid(*this),
				edges[ei].switch_vertex(*this).vid(*this),
			}};

			for (int vi = 0; vi < vids.size(); ++vi)
			{
				elements[ei][vi] = 2 * ei + vi;
				rest_positions.row(elements[ei][vi]) = vertex_attrs[vids[vi]].rest_position;
				displacements.row(elements[ei][vi]) = vertex_attrs[vids[vi]].displacement();
				energy_ranks(elements[ei][vi]) = int(edge_attr(edges[ei].eid(*this)).energy_rank);
				elastic_energy_ranks(elements[ei][vi]) = int(elastic_ranks.at(vids));
				contact_energy_ranks(elements[ei][vi]) = int(contact_ranks.at(vids));
			}
		}

		const double t0 = state.args["time"]["t0"];
		const double dt = state.args["time"]["dt"];
		const double save_dt = dt / 3;
		// current_time = t0 + t * dt for t = 1, 2, 3, ...
		// t = (current_time - t0) / dt
		const int time_steps = int(round((current_time - t0) / save_dt));

		// 0 -> 0 * dt + save_dt
		// 1 -> 1 * dt + save_dt
		// 2 -> 2 * dt + save_dt

		const auto vtu_name = [&](int i) -> std::string {
			return state.resolve_output_path(fmt::format("edge_ranks_{:d}.vtu", i));
		};

		io::VTUWriter writer;
		writer.add_field("displacement", displacements);
		writer.add_field("elastic_energy_rank", elastic_energy_ranks);
		writer.add_field("contact_energy_rank", contact_energy_ranks);
		writer.add_field("energy_rank", energy_ranks);
		writer.write_mesh(vtu_name(time_steps - 3), rest_positions, elements, /*is_simplicial=*/true);

		writer.add_field("displacement", displacements);
		writer.add_field("elastic_energy_rank", elastic_energy_ranks);
		writer.add_field("contact_energy_rank", contact_energy_ranks);
		writer.add_field("energy_rank", energy_ranks);
		writer.write_mesh(vtu_name(time_steps - 2), rest_positions, elements, /*is_simplicial=*/true);

		writer.add_field("displacement", displacements);
		writer.add_field("elastic_energy_rank", elastic_energy_ranks);
		writer.add_field("contact_energy_rank", contact_energy_ranks);
		writer.add_field("energy_rank", energy_ranks);
		writer.write_mesh(vtu_name(time_steps - 1), rest_positions, elements, /*is_simplicial=*/true);

		state.out_geom.save_pvd(
			state.resolve_output_path("edge_ranks.pvd"), vtu_name,
			time_steps - 1, /*t0=*/save_dt, save_dt);
	}

	// -------------------------------------------------------------------------
	// Template specializations

	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh
