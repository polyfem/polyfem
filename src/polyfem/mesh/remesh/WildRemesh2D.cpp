#include "WildRemesh2D.hpp"

#include <polyfem/io/OBJWriter.hpp>

#include <wmtk/utils/TupleUtils.hpp>

#include <igl/predicates/predicates.h>
#include <igl/boundary_facets.h>

#define VERTEX_ATTRIBUTE_GETTER(name, attribute)                                           \
	Eigen::MatrixXd WildRemeshing2D::name() const                                          \
	{                                                                                      \
		Eigen::MatrixXd attributes = Eigen::MatrixXd::Constant(vert_capacity(), DIM, NaN); \
		for (const Tuple &t : get_vertices())                                              \
			attributes.row(t.vid(*this)) = vertex_attrs[t.vid(*this)].attribute;           \
		return attributes;                                                                 \
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
		const EdgeMap<int> &edge_to_boundary_id,
		const std::vector<int> &body_ids)
	{
		assert(triangles.size() > 0);

		// --------------------------------------------------------------------
		// Determine which vertices are on the boundary

		// Partition mesh by body_ids
		assert(body_ids.size() == triangles.rows());
		std::unordered_map<int, std::vector<int>> body_triangles;
		for (int i = 0; i < triangles.rows(); ++i)
		{
			const int body_id = body_ids[i];
			if (body_triangles.find(body_id) == body_triangles.end())
				body_triangles[body_id] = std::vector<int>();
			body_triangles[body_id].push_back(i);
		}

		// Determine boundary vertices
		std::vector<bool> is_boundary_vertex(positions.rows(), false);
		for (const auto &[body_id, rows] : body_triangles)
		{
			Eigen::MatrixXi boundary_edges;
			igl::boundary_facets(triangles(rows, Eigen::all), boundary_edges);

			for (int i = 0; i < boundary_edges.rows(); ++i)
			{
				is_boundary_vertex[boundary_edges(i, 0)] = true;
				is_boundary_vertex[boundary_edges(i, 1)] = true;
			}
		}

		// --------------------------------------------------------------------

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
		wmtk::TriMesh::create_mesh(positions.rows(), tri);

		// Save the vertex position in the vertex attributes
		set_rest_positions(rest_positions);
		set_positions(positions);
		set_projected_quantities(projection_quantities);
		set_fixed(is_boundary_vertex);

		set_boundary_ids(edge_to_boundary_id);
		set_body_ids(body_ids);
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
			Eigen::MatrixXd::Constant(DIM * vert_capacity(), n_quantities, NaN);

		for (const Tuple &t : get_vertices())
		{
			const int vi = t.vid(*this);
			projected_quantities.middleRows<DIM>(DIM * vi) = vertex_attrs[vi].projection_quantities;
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
		assert(projected_quantities.rows() == DIM * vert_capacity());
		n_quantities = projected_quantities.cols();

		for (const Tuple &t : get_vertices())
		{
			const int vi = t.vid(*this);
			vertex_attrs[vi].projection_quantities =
				projected_quantities.middleRows<DIM>(DIM * vi);
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

	void WildRemeshing2D::write_obj(const std::string &path, bool deformed) const
	{
		io::OBJWriter::write(path, deformed ? positions() : rest_positions(), triangles());
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
			   && deformed_orientation != igl::predicates::Orientation::POSITIVE;
	}

	bool WildRemeshing2D::invariants(const std::vector<Tuple> &new_tris)
	{
		for (auto &t : new_tris)
		{
			if (is_inverted(t))
			{
				return false;
			}
		}
		return true;
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

} // namespace polyfem::mesh