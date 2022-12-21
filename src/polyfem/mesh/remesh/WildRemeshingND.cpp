#include "WildRemeshingND.hpp"

#include <polyfem/utils/GeometryUtils.hpp>

#define VERTEX_ATTRIBUTE_GETTER(name, attribute)                                                     \
	template <class WMTKMesh>                                                                        \
	Eigen::MatrixXd WildRemeshingND<WMTKMesh>::name() const                                          \
	{                                                                                                \
		Eigen::MatrixXd attributes = Eigen::MatrixXd::Constant(WMTKMesh::vert_capacity(), DIM, NaN); \
		for (const Tuple &t : WMTKMesh::get_vertices())                                              \
			attributes.row(t.vid(*this)) = vertex_attrs[t.vid(*this)].attribute;                     \
		return attributes;                                                                           \
	}

#define VERTEX_ATTRIBUTE_SETTER(name, attribute)                                 \
	template <class WMTKMesh>                                                    \
	void WildRemeshingND<WMTKMesh>::name(const Eigen::MatrixXd &attributes)      \
	{                                                                            \
		for (const Tuple &t : WMTKMesh::get_vertices())                          \
			vertex_attrs[t.vid(*this)].attribute = attributes.row(t.vid(*this)); \
	}

namespace polyfem::mesh
{
	static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

	template <class WMTKMesh>
	void WildRemeshingND<WMTKMesh>::init(
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXi &triangles,
		const Eigen::MatrixXd &projection_quantities,
		const BoundaryMap<int> &boundary_to_id,
		const std::vector<int> &body_ids)
	{
		WildRemeshing::init(rest_positions, positions, triangles, projection_quantities, boundary_to_id, body_ids);

		total_volume = 0;
		for (const Tuple &t : get_elements())
			total_volume += element_volume(t);
	}

	// -------------------------------------------------------------------------
	// Getters

	template <class WMTKMesh>
	std::vector<typename WildRemeshingND<WMTKMesh>::Tuple> WildRemeshingND<WMTKMesh>::get_elements() const
	{
		if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
			return WMTKMesh::get_faces();
		else
			return WMTKMesh::get_tets();
	}

	VERTEX_ATTRIBUTE_GETTER(rest_positions, rest_position)
	VERTEX_ATTRIBUTE_GETTER(positions, position)
	VERTEX_ATTRIBUTE_GETTER(displacements, displacement())

	template <class WMTKMesh>
	Eigen::MatrixXi WildRemeshingND<WMTKMesh>::edges() const
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
	Eigen::MatrixXi WildRemeshingND<WMTKMesh>::elements() const
	{
		const std::vector<Tuple> elements = get_elements();
		Eigen::MatrixXi F = Eigen::MatrixXi::Constant(elements.size(), dim() + 1, -1);
		for (size_t i = 0; i < elements.size(); i++)
		{
			std::array<size_t, DIM + 1> vids;
			if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
				vids = WMTKMesh::oriented_tri_vids(elements[i]);
			else
				vids = WMTKMesh::oriented_tet_vids(elements[i]);

			for (int j = 0; j < vids.size(); j++)
			{
				F(i, j) = vids[j];
			}
		}
		return F;
	}

	template <class WMTKMesh>
	Eigen::MatrixXd WildRemeshingND<WMTKMesh>::projected_quantities() const
	{
		Eigen::MatrixXd projected_quantities =
			Eigen::MatrixXd::Constant(dim() * WMTKMesh::vert_capacity(), n_quantities(), NaN);

		for (const Tuple &t : WMTKMesh::get_vertices())
		{
			const int vi = t.vid(*this);
			projected_quantities.middleRows(dim() * vi, dim()) = vertex_attrs[vi].projection_quantities;
		}

		return projected_quantities;
	}

	template <>
	WildRemeshingND<wmtk::TriMesh>::BoundaryMap<int> WildRemeshingND<wmtk::TriMesh>::boundary_ids() const
	{
		const std::vector<Tuple> edges = get_edges();
		EdgeMap<int> boundary_ids;
		for (const Tuple &edge : edges)
		{
			size_t e0 = edge.vid(*this);
			size_t e1 = edge.switch_vertex(*this).vid(*this);
			if (e1 < e0)
				std::swap(e0, e1);
			boundary_ids[std::make_pair(e0, e1)] = boundary_attrs[edge.eid(*this)].boundary_id;
		}
		return boundary_ids;
	}

	template <>
	WildRemeshingND<wmtk::TetMesh>::BoundaryMap<int> WildRemeshingND<wmtk::TetMesh>::boundary_ids() const
	{
		const std::vector<Tuple> faces = get_faces();
		FaceMap<int> boundary_ids;
		for (const Tuple &face : faces)
		{
			const size_t f0 = face.vid(*this);
			const size_t f1 = face.switch_vertex(*this).vid(*this);
			const size_t f2 = face.switch_edge(*this).switch_vertex(*this).vid(*this);

			std::vector<size_t> f_ids = {f0, f1, f2};
			std::sort(f_ids.begin(), f_ids.end());

			boundary_ids[f_ids] = boundary_attrs[face.fid(*this)].boundary_id;
		}
		return boundary_ids;
	}

	template <class WMTKMesh>
	std::vector<int> WildRemeshingND<WMTKMesh>::body_ids() const
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
	std::vector<int> WildRemeshingND<WMTKMesh>::boundary_nodes() const
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
	void WildRemeshingND<WMTKMesh>::set_projected_quantities(
		const Eigen::MatrixXd &projected_quantities)
	{
		assert(projected_quantities.rows() == dim() * WMTKMesh::vert_capacity());
		m_n_quantities = projected_quantities.cols();

		for (const Tuple &t : WMTKMesh::get_vertices())
		{
			const int vi = t.vid(*this);
			vertex_attrs[vi].projection_quantities =
				projected_quantities.middleRows(dim() * vi, dim());
		}
	}

	template <class WMTKMesh>
	void WildRemeshingND<WMTKMesh>::set_fixed(
		const std::vector<bool> &fixed)
	{
		assert(fixed.size() == WMTKMesh::vert_capacity());
		for (const Tuple &t : WMTKMesh::get_vertices())
			vertex_attrs[t.vid(*this)].fixed = fixed[t.vid(*this)];
	}

	template <class WMTKMesh>
	void WildRemeshingND<WMTKMesh>::set_boundary_ids(
		const BoundaryMap<int> &boundary_to_id)
	{
		if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
		{
			assert(std::holds_alternative<EdgeMap<int>>(boundary_to_id));
			const EdgeMap<int> &edge_to_boundary_id = std::get<EdgeMap<int>>(boundary_to_id);
			for (const Tuple &edge : WMTKMesh::get_edges())
			{
				size_t e0 = edge.vid(*this);
				size_t e1 = edge.switch_vertex(*this).vid(*this);
				if (e1 < e0)
					std::swap(e0, e1);
				boundary_attrs[edge.eid(*this)].boundary_id = edge_to_boundary_id.at(std::make_pair(e0, e1));
			}
		}
		else
		{
			assert(std::holds_alternative<FaceMap<int>>(boundary_to_id));
			const FaceMap<int> &face_to_boundary_id = std::get<FaceMap<int>>(boundary_to_id);
			for (const Tuple &face : WMTKMesh::get_faces())
			{
				size_t f0 = face.vid(*this);
				size_t f1 = face.switch_vertex(*this).vid(*this);
				size_t f2 = face.switch_edge(*this).switch_vertex(*this).vid(*this);
				std::vector<size_t> f_ids = {{f0, f1, f2}};
				std::sort(f_ids.begin(), f_ids.end());
				boundary_attrs[face.fid(*this)].boundary_id = face_to_boundary_id.at(f_ids);
			}
		}
	}

	template <class WMTKMesh>
	void WildRemeshingND<WMTKMesh>::set_body_ids(
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
	bool WildRemeshingND<WMTKMesh>::invariants(const std::vector<Tuple> &new_tris)
	{
		// for (auto &t : new_tris)
		for (auto &t : get_elements())
		{
			if (is_inverted(t))
			{
				log_and_throw_error("Inverted triangle found, invariants violated");
				return false;
			}
		}
		return true;
	}

	// -------------------------------------------------------------------------
	// Utils

	template <class WMTKMesh>
	double WildRemeshingND<WMTKMesh>::edge_length(const Tuple &e) const
	{
		const auto &e0 = vertex_attrs[e.vid(*this)].position;
		const auto &e1 = vertex_attrs[e.switch_vertex(*this).vid(*this)].position;
		return (e1 - e0).norm();
	}

	template <class WMTKMesh>
	double WildRemeshingND<WMTKMesh>::element_volume(const Tuple &e) const
	{
		if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
		{
			const std::array<size_t, 3> vids = WMTKMesh::oriented_tri_vids(e);
			return utils::triangle_area_2D(
				vertex_attrs[vids[0]].rest_position,
				vertex_attrs[vids[1]].rest_position,
				vertex_attrs[vids[2]].rest_position);
		}
		else
		{
			const std::array<size_t, 4> vids = WMTKMesh::oriented_tet_vids(e);
			return utils::tetrahedron_volume(
				vertex_attrs[vids[0]].rest_position,
				vertex_attrs[vids[1]].rest_position,
				vertex_attrs[vids[2]].rest_position,
				vertex_attrs[vids[3]].rest_position);
		}
	}

	// -------------------------------------------------------------------------
	// Template specializations

	template class WildRemeshingND<wmtk::TriMesh>;
	template class WildRemeshingND<wmtk::TetMesh>;

} // namespace polyfem::mesh
