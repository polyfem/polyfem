#include "WildRemesh2D.hpp"

#include <polyfem/mesh/remesh/wild_remesh/AMIPSForm.hpp>
#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <igl/boundary_facets.h>
#include <igl/predicates/predicates.h>

#define VERTEX_ATTRIBUTE_GETTER(name, attribute)                                 \
	Eigen::MatrixXd WildRemeshing2D::name() const                                \
	{                                                                            \
		Eigen::MatrixXd attributes(vert_capacity(), dim());                      \
		for (const Tuple &t : get_vertices())                                    \
			attributes.row(t.vid(*this)) = vertex_attrs[t.vid(*this)].attribute; \
		return attributes;                                                       \
	}

#define VERTEX_ATTRIBUTE_SETTER(name, attribute)                                 \
	void WildRemeshing2D::name(const Eigen::MatrixXd &attributes)                \
	{                                                                            \
		for (const Tuple &t : get_vertices())                                    \
			vertex_attrs[t.vid(*this)].attribute = attributes.row(t.vid(*this)); \
	}

namespace polyfem::mesh
{
	void WildRemeshing2D::create_mesh(
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXd &velocities,
		const Eigen::MatrixXd &accelerations,
		const Eigen::MatrixXi &triangles)
	{
		Eigen::MatrixXi boundary_edges;
		igl::boundary_facets(triangles, boundary_edges);

		std::vector<bool> is_boundary_vertex(positions.rows(), false);
		for (int i = 0; i < boundary_edges.rows(); ++i)
		{
			is_boundary_vertex[boundary_edges(i, 0)] = true;
			is_boundary_vertex[boundary_edges(i, 1)] = true;
		}

		// Register attributes
		p_vertex_attrs = &vertex_attrs;

		// Convert from eigen to internal representation (TODO: move to utils and remove it from all app)
		std::vector<std::array<size_t, 3>> tri(triangles.rows());

		for (int i = 0; i < triangles.rows(); i++)
			for (int j = 0; j < 3; j++)
				tri[i][j] = (size_t)triangles(i, j);

		// Initialize the trimesh class which handles connectivity
		wmtk::TriMesh::create_mesh(positions.rows(), tri);

		// Save the vertex position in the vertex attributes
		for (unsigned i = 0; i < positions.rows(); ++i)
		{
			vertex_attrs[i].rest_position = rest_positions.row(i).head(dim());
			vertex_attrs[i].position = positions.row(i).head(dim());
			vertex_attrs[i].velocity = velocities.row(i).head(dim());
			vertex_attrs[i].acceleration = accelerations.row(i).head(dim());
			vertex_attrs[i].frozen = is_boundary_vertex[i];
		}
	}

	VERTEX_ATTRIBUTE_GETTER(rest_positions, rest_position)
	VERTEX_ATTRIBUTE_GETTER(positions, position)
	VERTEX_ATTRIBUTE_GETTER(displacements, displacement())
	VERTEX_ATTRIBUTE_GETTER(velocities, velocity)
	VERTEX_ATTRIBUTE_GETTER(accelerations, acceleration)

	VERTEX_ATTRIBUTE_SETTER(set_positions, position)
	VERTEX_ATTRIBUTE_SETTER(set_velocities, velocity)
	VERTEX_ATTRIBUTE_SETTER(set_accelerations, acceleration)

	Eigen::MatrixXi WildRemeshing2D::triangles() const
	{
		MatrixXi triangles(tri_capacity(), 3);
		for (const Tuple &t : get_faces())
		{
			const size_t i = t.fid(*this);
			const std::array<Tuple, 3> vs = oriented_tri_vertices(t);
			for (int j = 0; j < 3; j++)
			{
				triangles(i, j) = vs[j].vid(*this);
			}
		}
		return triangles;
	}

	void WildRemeshing2D::write_obj(const std::string &path, bool deformed) const
	{
		io::OBJWriter::write(path, deformed ? positions() : rest_positions(), triangles());
	}

	double WildRemeshing2D::compute_global_energy() const
	{
		double energy = 0;
		for (const Tuple &t : get_faces())
		{
			// Global ids of the vertices of the triangle
			const std::array<size_t, 3> its = super::oriented_tri_vids(t);
			// Energy evaluation
			energy += solver::AMIPSForm::energy(
				vertex_attrs[its[0]].rest_position,
				vertex_attrs[its[1]].rest_position,
				vertex_attrs[its[2]].rest_position,
				vertex_attrs[its[0]].position,
				vertex_attrs[its[1]].position,
				vertex_attrs[its[2]].position);
		}
		return energy;
	}

	bool WildRemeshing2D::is_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<Tuple, 3> vs = oriented_tri_vertices(loc);

		igl::predicates::exactinit();

		// Use igl for checking orientation
		igl::predicates::Orientation res = igl::predicates::orient2d(
			vertex_attrs[vs[0].vid(*this)].rest_position,
			vertex_attrs[vs[1].vid(*this)].rest_position,
			vertex_attrs[vs[2].vid(*this)].rest_position);

		// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
		return (res != igl::predicates::Orientation::POSITIVE);
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

	void WildRemeshing2D::update_positions()
	{
		// Assume the rest positions and triangles have been updated
		const Eigen::MatrixXd proposed_rest_positions = rest_positions();
		const Eigen::MatrixXi proposed_triangles = triangles();

		// Save old values
		const int old_n_bases = state_.n_bases;
		const std::vector<ElementBases> old_bases = state_.bases;
		const std::vector<ElementBases> old_geom_bases = state_.geom_bases();

		// Old values of independent variables
		Eigen::MatrixXd y(rest_positions_before.size(), 3);
		y.col(0) = utils::flatten(positions_before - rest_positions_before);
		y.col(1) = utils::flatten(velocities_before);
		y.col(2) = utils::flatten(accelerations_before);

		// --------------------------------------------------------------------

		// TODO: All I need out of this is the new bases (no need to modify the state)
		state_.load_mesh(proposed_rest_positions, proposed_triangles);
		// FIXME:
		state_.mesh->compute_boundary_ids(1e-6);
		state_.mesh->set_body_ids(std::vector<int>(state_.mesh->n_elements(), 1));
		state_.set_materials(); // TODO: Explain why I need this?
		state_.build_basis();
		state_.assemble_rhs();
		state_.assemble_stiffness_mat();
		std::shared_ptr<assembler::RhsAssembler> rhs_assembler = state_.build_rhs_assembler();

		// --------------------------------------------------------------------

		// L2 Projection
		Eigen::MatrixXd x;
		L2_projection(
			state_, *rhs_assembler, /*is_volume=*/false, /*size=*/dim(),
			old_n_bases, old_bases, old_geom_bases,            // from
			state_.n_bases, state_.bases, state_.geom_bases(), // to
			y, x, time_, /*lump_mass_matrix=*/false);

		set_positions(proposed_rest_positions + utils::unflatten(x.col(0), dim()));
		set_velocities(utils::unflatten(x.col(1), dim()));
		set_accelerations(utils::unflatten(x.col(2), dim()));

		write_rest_obj("proposed_rest_mesh.obj");
		write_deformed_obj("proposed_deformed_mesh.obj");
	}

} // namespace polyfem::mesh