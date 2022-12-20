#include "WildRemesh2D.hpp"

#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/VTUWriter.hpp>

#include <wmtk/utils/TupleUtils.hpp>

#include <igl/predicates/predicates.h>
#include <igl/boundary_facets.h>

#define VERTEX_ATTRIBUTE_GETTER(name, attribute)                                             \
	Eigen::MatrixXd WildRemeshing2D::name() const                                            \
	{                                                                                        \
		Eigen::MatrixXd attributes = Eigen::MatrixXd::Constant(vert_capacity(), dim(), NaN); \
		for (const Tuple &t : get_vertices())                                                \
			attributes.row(t.vid(*this)) = vertex_attrs[t.vid(*this)].attribute;             \
		return attributes;                                                                   \
	}

#define VERTEX_ATTRIBUTE_SETTER(name, attribute)                                 \
	void WildRemeshing2D::name(const Eigen::MatrixXd &attributes)                \
	{                                                                            \
		for (const Tuple &t : get_vertices())                                    \
			vertex_attrs[t.vid(*this)].attribute = attributes.row(t.vid(*this)); \
	}

namespace polyfem::mesh
{
	static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

	void WildRemeshing2D::init(
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXi &triangles,
		const Eigen::MatrixXd &projection_quantities,
		const EdgeMap<int> &edge_to_boundary_id, // TODO: this has to change for 3D
		const std::vector<int> &body_ids)
	{
		WildRemeshing::init(rest_positions, positions, triangles, projection_quantities, edge_to_boundary_id, body_ids);

		total_area = 0;
		for (const Tuple &t : get_faces())
			total_area += triangle_area(t);
	}

	void WildRemeshing2D::create_mesh(const size_t num_vertices, const Eigen::MatrixXi &triangles)
	{
		// Register attributes
		p_vertex_attrs = &vertex_attrs;
		p_edge_attrs = &edge_attrs;
		p_face_attrs = &face_attrs;

		// Convert from eigen to internal representation
		std::vector<std::array<size_t, 3>> tri(triangles.rows());
		for (int i = 0; i < triangles.rows(); i++)
			for (int j = 0; j < 3; j++)
				tri[i][j] = (size_t)triangles(i, j);

		// Initialize the trimesh class which handles connectivity
		wmtk::TriMesh::create_mesh(num_vertices, tri);
	}

	// ========================================================================
	// Getters

	VERTEX_ATTRIBUTE_GETTER(rest_positions, rest_position)
	VERTEX_ATTRIBUTE_GETTER(positions, position)
	VERTEX_ATTRIBUTE_GETTER(displacements, displacement())

	Eigen::MatrixXi WildRemeshing2D::edges() const
	{
		const std::vector<Tuple> edges = get_edges();
		Eigen::MatrixXi E = Eigen::MatrixXi::Constant(edges.size(), 2, -1);
		for (int i = 0; i < edges.size(); ++i)
		{
			const Tuple &e = edges[i];
			E(i, 0) = e.vid(*this);
			E(i, 1) = e.switch_vertex(*this).vid(*this);
		}
		return E;
	}

	Eigen::MatrixXi WildRemeshing2D::triangles() const
	{
		const std::vector<Tuple> faces = get_faces();
		Eigen::MatrixXi triangles = Eigen::MatrixXi::Constant(faces.size(), 3, -1);
		for (size_t i = 0; i < faces.size(); i++)
		{
			const Tuple &t = faces[i];
			const std::array<Tuple, 3> vs = oriented_tri_vertices(t);
			for (int j = 0; j < 3; j++)
			{
				triangles(i, j) = vs[j].vid(*this);
			}
		}
		return triangles;
	}

	Eigen::MatrixXd WildRemeshing2D::projected_quantities() const
	{
		Eigen::MatrixXd projected_quantities =
			Eigen::MatrixXd::Constant(dim() * vert_capacity(), n_quantities(), NaN);

		for (const Tuple &t : get_vertices())
		{
			const int vi = t.vid(*this);
			projected_quantities.middleRows(dim() * vi, dim()) = vertex_attrs[vi].projection_quantities;
		}

		return projected_quantities;
	}

	WildRemeshing2D::EdgeMap<int> WildRemeshing2D::boundary_ids() const
	{
		const std::vector<Tuple> edges = get_edges();
		EdgeMap<int> boundary_ids;
		for (int i = 0; i < edges.size(); ++i)
		{
			size_t e0 = edges[i].vid(*this);
			size_t e1 = edges[i].switch_vertex(*this).vid(*this);
			if (e1 < e0)
				std::swap(e0, e1);
			boundary_ids[std::make_pair(e0, e1)] = edge_attrs[edges[i].eid(*this)].boundary_id;
		}
		return boundary_ids;
	}

	std::vector<int> WildRemeshing2D::body_ids() const
	{
		const std::vector<Tuple> faces = get_faces();
		std::vector<int> body_ids(faces.size(), -1);
		for (size_t i = 0; i < faces.size(); i++)
		{
			const Tuple &t = faces[i];
			body_ids[i] = face_attrs[t.fid(*this)].body_id;
		}
		return body_ids;
	}

	// ========================================================================
	// Setters

	VERTEX_ATTRIBUTE_SETTER(set_rest_positions, rest_position)
	VERTEX_ATTRIBUTE_SETTER(set_positions, position)

	void WildRemeshing2D::set_fixed(const std::vector<bool> &fixed)
	{
		assert(fixed.size() == vert_capacity());
		for (const Tuple &t : get_vertices())
			vertex_attrs[t.vid(*this)].fixed = fixed[t.vid(*this)];
	}

	void WildRemeshing2D::set_projected_quantities(const Eigen::MatrixXd &projected_quantities)
	{
		assert(projected_quantities.rows() == dim() * vert_capacity());
		m_n_quantities = projected_quantities.cols();

		for (const Tuple &t : get_vertices())
		{
			const int vi = t.vid(*this);
			vertex_attrs[vi].projection_quantities =
				projected_quantities.middleRows(dim() * vi, dim());
		}
	}

	void WildRemeshing2D::set_boundary_ids(const EdgeMap<int> &edge_to_boundary_id)
	{
		for (const Tuple &e : get_edges())
		{
			size_t e0 = e.vid(*this);
			size_t e1 = e.switch_vertex(*this).vid(*this);
			if (e1 < e0)
				std::swap(e0, e1);
			edge_attrs[e.eid(*this)].boundary_id = edge_to_boundary_id.at(std::make_pair(e0, e1));
		}
	}

	void WildRemeshing2D::set_body_ids(const std::vector<int> &body_ids)
	{
		for (const Tuple &f : get_faces())
		{
			face_attrs[f.fid(*this)].body_id = body_ids.at(f.fid(*this));
		}
	}

	// ========================================================================

	bool WildRemeshing2D::is_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<Tuple, 3> vs = oriented_tri_vertices(loc);

		igl::predicates::exactinit();

		// Use igl for checking orientation
		igl::predicates::Orientation rest_orientation = igl::predicates::orient2d(
			vertex_attrs[vs[0].vid(*this)].rest_position,
			vertex_attrs[vs[1].vid(*this)].rest_position,
			vertex_attrs[vs[2].vid(*this)].rest_position);
		igl::predicates::Orientation deformed_orientation = igl::predicates::orient2d(
			vertex_attrs[vs[0].vid(*this)].position,
			vertex_attrs[vs[1].vid(*this)].position,
			vertex_attrs[vs[2].vid(*this)].position);

		// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
		return rest_orientation != igl::predicates::Orientation::POSITIVE
			   || deformed_orientation != igl::predicates::Orientation::POSITIVE;
	}

	bool WildRemeshing2D::invariants(const std::vector<Tuple> &new_tris)
	{
		// for (auto &t : new_tris)
		for (auto &t : get_faces())
		{
			if (is_inverted(t))
			{
				log_and_throw_error("Inverted triangle found, invariants violated");
				return false;
			}
		}
		return true;
	}

	// ========================================================================

	double WildRemeshing2D::triangle_area(const Tuple &triangle) const
	{
		const std::array<WildRemeshing2D::Tuple, 3> vs = oriented_tri_vertices(triangle);
		return utils::triangle_area_2D(
			vertex_attrs[vs[0].vid(*this)].rest_position,
			vertex_attrs[vs[1].vid(*this)].rest_position,
			vertex_attrs[vs[2].vid(*this)].rest_position);
	}

	// ========================================================================

	std::vector<WildRemeshing2D::Tuple> WildRemeshing2D::new_edges_after(
		const std::vector<Tuple> &tris) const
	{
		std::vector<Tuple> new_edges;

		for (auto t : tris)
		{
			for (auto j = 0; j < 3; j++)
			{
				new_edges.push_back(tuple_from_edge(t.fid(*this), j));
			}
		}
		wmtk::unique_edge_tuples(*this, new_edges);
		return new_edges;
	}

	// ========================================================================

	std::vector<int> WildRemeshing2D::boundary_nodes() const
	{
		std::vector<int> boundary_nodes;
		for (const Tuple &t : get_vertices())
			if (vertex_attrs[t.vid(*this)].fixed)
				boundary_nodes.push_back(t.vid(*this));
		return boundary_nodes;
	}

	std::vector<WildRemeshing2D::Tuple> WildRemeshing2D::boundary_edges() const
	{
		std::vector<Tuple> boundary_edges;
		for (const Tuple &e : get_edges())
			if (!e.switch_face(*this))
				boundary_edges.push_back(e);
		return boundary_edges;
	}

} // namespace polyfem::mesh