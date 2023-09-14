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
	rank_edges(const Remesher::EdgeMap<double> &edge_energy, const json &args)
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

		const double split_threshold = args["split"]["culling_threshold"];
		const double split_tolerance = args["split"]["acceptance_tolerance"];
		const double top_energy_threshold = (max_energy - min_energy) * split_threshold + min_energy;
		const double top_element_threshold = sorted_energies[int(sorted_energies.size() * split_threshold)];
		const double top_threshold = std::max(std::min(top_energy_threshold, top_element_threshold), std::min(1e-12, split_tolerance));

		const double collapse_threshold = args["collapse"]["culling_threshold"];
		const double bottom_energy_threshold = (max_energy - min_energy) * collapse_threshold + min_energy;
		// const double bottom_element_threshold = sorted_energies[int(sorted_energies.size() * collapse_threshold)];
		const double bottom_threshold = bottom_energy_threshold;

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

		const auto edge_elastic_ranks = rank_edges<WMTKMesh>(elastic_energy, args);
		const auto edge_contact_ranks = rank_edges<WMTKMesh>(contact_energy, args);

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

		// write_edge_ranks_mesh(edge_elastic_ranks, edge_contact_ranks);
	}

	// -------------------------------------------------------------------------
	// Getters

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

		const std::unordered_map<int, std::array<bool, 3>> bcs =
			state.boundary_conditions_ids("dirichlet_boundary");
		std::vector<int> boundary_ids;
		const std::vector<Tuple> boundary_facets = this->boundary_facets(&boundary_ids);
		for (int i = 0; i < boundary_facets.size(); ++i)
		{
			const Tuple &t = boundary_facets[i];
			const auto bc = bcs.find(boundary_ids[i]);

			if (bc == bcs.end())
				continue;

			for (int d = 0; d < this->dim(); ++d)
			{
				if (!bc->second[d])
					continue;

				for (const size_t vid : facet_vids(t))
					boundary_nodes.push_back(dim() * vertex_to_basis[vid] + d);
			}
		}

		// Sort and remove the duplicate boundary_nodes.
		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto new_end = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.erase(new_end, boundary_nodes.end());

		return boundary_nodes;
	}

	template <class WMTKMesh>
	std::vector<typename WMTKMesh::Tuple>
	WildRemesher<WMTKMesh>::boundary_facets(std::vector<int> *boundary_ids) const
	{
		POLYFEM_REMESHER_SCOPED_TIMER("boundary_facets");

		size_t element_capacity;
		if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
			element_capacity = WMTKMesh::tri_capacity();
		else
			element_capacity = WMTKMesh::tet_capacity();

		const size_t max_tid = std::min(boundary_attrs.size() / FACETS_PER_ELEMENT, element_capacity);

		std::vector<Tuple> boundary_facets;
		for (int tid = 0; tid < max_tid; ++tid)
		{
			const Tuple t = tuple_from_element(tid);
			if (!t.is_valid(*this))
				continue;

			for (int local_fid = 0; local_fid < FACETS_PER_ELEMENT; ++local_fid)
			{
				const Tuple facet_tuple = tuple_from_facet(tid, local_fid);
				const int boundary_id = boundary_attrs[facet_id(facet_tuple)].boundary_id;
				assert((boundary_id >= 0) == is_boundary_facet(facet_tuple));
				if (boundary_id >= 0)
				{
					boundary_facets.push_back(facet_tuple);
					if (boundary_ids)
						boundary_ids->push_back(boundary_id);
				}
			}
		}

#ifndef NDEBUG
		{
			std::vector<size_t> boundary_facet_ids;
			for (const Tuple &f : boundary_facets)
				boundary_facet_ids.push_back(facet_id(f));
			std::sort(boundary_facet_ids.begin(), boundary_facet_ids.end());

			std::vector<size_t> gt_boundary_facet_ids;
			for (const Tuple &f : get_facets())
				if (is_boundary_facet(f))
					gt_boundary_facet_ids.push_back(facet_id(f));
			std::sort(gt_boundary_facet_ids.begin(), gt_boundary_facet_ids.end());

			assert(boundary_facets.size() == gt_boundary_facet_ids.size());
			assert(boundary_facet_ids == gt_boundary_facet_ids);
		}
#endif

		return boundary_facets;
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

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::set_boundary_ids(const BoundaryMap<int> &boundary_to_id)
	{
		for (const Tuple &face : get_facets())
		{
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
			{
				assert(std::holds_alternative<EdgeMap<int>>(boundary_to_id));
				const EdgeMap<int> &edge_to_boundary_id = std::get<EdgeMap<int>>(boundary_to_id);
				boundary_attrs[facet_id(face)].boundary_id = edge_to_boundary_id.at(facet_vids(face));
			}
			else
			{
				assert(std::holds_alternative<FaceMap<int>>(boundary_to_id));
				const FaceMap<int> &face_to_boundary_id = std::get<FaceMap<int>>(boundary_to_id);
				boundary_attrs[facet_id(face)].boundary_id = face_to_boundary_id.at(facet_vids(face));
			}

#ifndef NDEBUG
			if (is_boundary_facet(face))
				assert(boundary_attrs[facet_id(face)].boundary_id >= 0);
			else
				assert(boundary_attrs[facet_id(face)].boundary_id == -1);
#endif
		}
	}

	// -------------------------------------------------------------------------

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::invariants(const std::vector<Tuple> &new_tris)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("WildRemesher::invariants");
		// for (auto &t : new_tris)
		for (auto &t : get_elements())
		{
			if (is_inverted(t))
			{
				static int inversion_cnt = 0;
				write_mesh(state.resolve_output_path(fmt::format("inversion_{:04d}.vtu", inversion_cnt++)));
				log_and_throw_error("Inverted element found, invariants violated!");
				return false;
			}
		}
		return true;
	}

	// -------------------------------------------------------------------------
	// Utils

	template <class WMTKMesh>
	double WildRemesher<WMTKMesh>::rest_edge_length(const Tuple &e) const
	{
		const auto &e0 = vertex_attrs[e.vid(*this)].rest_position;
		const auto &e1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position;
		return (e1 - e0).norm();
	}

	template <class WMTKMesh>
	double WildRemesher<WMTKMesh>::deformed_edge_length(const Tuple &e) const
	{
		const auto &e0 = vertex_attrs[e.vid(*this)].position;
		const auto &e1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].position;
		return (e1 - e0).norm();
	}

	template <class WMTKMesh>
	typename WildRemesher<WMTKMesh>::VectorNd
	WildRemesher<WMTKMesh>::rest_edge_center(const Tuple &e) const
	{
		const VectorNd &e0 = vertex_attrs[e.vid(*this)].rest_position;
		const VectorNd &e1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].rest_position;
		return (e1 + e0) / 2.0;
	}

	template <class WMTKMesh>
	typename WildRemesher<WMTKMesh>::VectorNd
	WildRemesher<WMTKMesh>::deformed_edge_center(const Tuple &e) const
	{
		const VectorNd &e0 = vertex_attrs[e.vid(*this)].position;
		const VectorNd &e1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].position;
		return (e1 + e0) / 2.0;
	}

	template <class WMTKMesh>
	std::vector<typename WMTKMesh::Tuple> WildRemesher<WMTKMesh>::get_edges_for_elements(
		const std::vector<Tuple> &elements) const
	{
		std::vector<Tuple> edges;
		for (auto t : elements)
			for (int j = 0; j < EDGES_PER_ELEMENT; ++j)
				edges.push_back(WMTKMesh::tuple_from_edge(element_id(t), j));
		wmtk::unique_edge_tuples(*this, edges);
		return edges;
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
	void WildRemesher<WMTKMesh>::element_aabb(const Tuple &t, polyfem::VectorNd &min, polyfem::VectorNd &max) const
	{
		min.setConstant(dim(), std::numeric_limits<double>::infinity());
		max.setConstant(dim(), -std::numeric_limits<double>::infinity());

		for (const size_t vid : element_vids(t))
		{
			min = min.cwiseMin(vertex_attrs[vid].rest_position);
			max = max.cwiseMax(vertex_attrs[vid].rest_position);
		}
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
				if (!contact_ranks.empty())
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

		paraviewo::VTUWriter writer;
		writer.add_field("displacement", displacements);
		writer.add_field("elastic_energy_rank", elastic_energy_ranks);
		if (!contact_ranks.empty())
			writer.add_field("contact_energy_rank", contact_energy_ranks);
		writer.add_field("energy_rank", energy_ranks);
		writer.write_mesh(vtu_name(time_steps - 3), rest_positions, elements, /*is_simplicial=*/true, /*has_poly=*/false);

		writer.add_field("displacement", displacements);
		writer.add_field("elastic_energy_rank", elastic_energy_ranks);
		if (!contact_ranks.empty())
			writer.add_field("contact_energy_rank", contact_energy_ranks);
		writer.add_field("energy_rank", energy_ranks);
		writer.write_mesh(vtu_name(time_steps - 2), rest_positions, elements, /*is_simplicial=*/true, /*has_poly=*/false);

		writer.add_field("displacement", displacements);
		writer.add_field("elastic_energy_rank", elastic_energy_ranks);
		if (!contact_ranks.empty())
			writer.add_field("contact_energy_rank", contact_energy_ranks);
		writer.add_field("energy_rank", energy_ranks);
		writer.write_mesh(vtu_name(time_steps - 1), rest_positions, elements, /*is_simplicial=*/true, /*has_poly=*/false);

		state.out_geom.save_pvd(
			state.resolve_output_path("edge_ranks.pvd"), vtu_name,
			time_steps - 1, /*t0=*/save_dt, save_dt);
	}

	// -------------------------------------------------------------------------
	// Template specializations

	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh
