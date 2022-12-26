#include "WildRemesher.hpp"

#include <polyfem/utils/GeometryUtils.hpp>

#include <wmtk/utils/TupleUtils.hpp>

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

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::init(
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXi &elements,
		const Eigen::MatrixXd &projection_quantities,
		const BoundaryMap<int> &boundary_to_id,
		const std::vector<int> &body_ids)
	{
		Remesher::init(rest_positions, positions, elements, projection_quantities, boundary_to_id, body_ids);

		total_volume = 0;
		for (const Tuple &t : get_elements())
			total_volume += element_volume(t);
		assert(total_volume > 0);

#ifndef NDEBUG
		assert(get_elements().size() == elements.rows());
		for (const Tuple &t : get_elements())
			assert(!is_inverted(t));
#endif
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
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				body_ids[i] = element_attrs[elements[i].fid(*this)].body_id;
			else
				body_ids[i] = element_attrs[elements[i].tid(*this)].body_id;
		}
		return body_ids;
	}

	template <class WMTKMesh>
	std::vector<int> WildRemesher<WMTKMesh>::boundary_nodes() const
	{
		std::vector<int> boundary_nodes;
		for (const Tuple &t : WMTKMesh::get_vertices())
			if (vertex_attrs[t.vid(*this)].fixed)
				boundary_nodes.push_back(t.vid(*this));
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
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				element_attrs[elements[i].fid(*this)].body_id = body_ids.at(i);
			else
				element_attrs[elements[i].tid(*this)].body_id = body_ids.at(i);
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
	std::vector<typename WMTKMesh::Tuple> WildRemesher<WMTKMesh>::new_edges_after(
		const std::vector<Tuple> &elements) const
	{
		constexpr int EDGES_IN_ELEMENT = [] {
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				return 3;
			else
				return 6;
		}();

		std::vector<Tuple> new_edges;
		for (const Tuple &t : elements)
		{
			for (auto j = 0; j < EDGES_IN_ELEMENT; j++)
			{
				new_edges.push_back(WMTKMesh::tuple_from_edge(element_id(t), j));
			}
		}
		wmtk::unique_edge_tuples(*this, new_edges);
		return new_edges;
	}

	// -------------------------------------------------------------------------
	// Template specializations

	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh
