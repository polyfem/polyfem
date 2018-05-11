////////////////////////////////////////////////////////////////////////////////
#include "Mesh.hpp"
#include "Mesh2D.hpp"
#include "Mesh3D.hpp"
#include "StringUtils.hpp"
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_geometry.h>
////////////////////////////////////////////////////////////////////////////////

namespace {

	bool is_planar(const GEO::Mesh &M) {
		if (M.vertices.dimension() == 2) {
			return true;
		}
		assert(M.vertices.dimension() == 3);
		GEO::vec3 min_corner, max_corner;
		GEO::get_bbox(M, &min_corner[0], &max_corner[0]);
		return (max_corner[2] - min_corner[2]) < 1e-5;
	}

} // anonymous namespace

std::unique_ptr<poly_fem::Mesh> poly_fem::Mesh::create(GEO::Mesh &meshin) {
	if (is_planar(meshin)) {
		auto mesh = std::make_unique<Mesh2D>();
		if (mesh->load(meshin)) {
			return mesh;
		}
	} else {
		auto mesh = std::make_unique<Mesh3D>();
		meshin.cells.connect();
		if (mesh->load(meshin)) {
			return mesh;
		}
	}

	std::cerr << "Failed to load mesh" << std::endl;
	return nullptr;
}

std::unique_ptr<poly_fem::Mesh> poly_fem::Mesh::create(const std::string &path) {
	std::string lowername = path;
	std::transform(lowername.begin(), lowername.end(), lowername.begin(), ::tolower);
	if (StringUtils::endswidth(lowername, ".hybrid")) {
		auto mesh = std::make_unique<Mesh3D>();
		if (mesh->load(path)){
			return mesh;
		}
	} else {
		GEO::Mesh tmp;
		if (GEO::mesh_load(path, tmp)) {
			return create(tmp);
		}
	}
	std::cerr << "Failed to load mesh: " << path << std::endl;
	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::Mesh::edge_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_edges(), dimension());
	for (int e = 0; e < n_edges(); ++e) {
		barycenters.row(e) = edge_barycenter(e);
	}
}

void poly_fem::Mesh::face_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_faces(), dimension());
	for (int f = 0; f < n_faces(); ++f) {
		barycenters.row(f) = face_barycenter(f);
	}
}

void poly_fem::Mesh::cell_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_cells(), dimension());
	for (int c = 0; c < n_cells(); ++c) {
		barycenters.row(c) = cell_barycenter(c);
	}
}

////////////////////////////////////////////////////////////////////////////////

//Queries on the tags
bool poly_fem::Mesh::is_spline_compatible(const int el_id) const
{
	if(is_volume()){
		return
		elements_tag_[el_id] == ElementType::RegularInteriorCube ||
		elements_tag_[el_id] == ElementType::RegularBoundaryCube; // ||
		// elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube ||
		// elements_tag_[el_id] == ElementType::SimpleSingularBoundaryCube;
	}
	else
	{
		return
		elements_tag_[el_id] == ElementType::RegularInteriorCube ||
		elements_tag_[el_id] == ElementType::RegularBoundaryCube; // ||
		// elements_tag_[el_id] == ElementType::InterfaceCube ||
		// elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube;
	}
}

// -----------------------------------------------------------------------------

bool poly_fem::Mesh::is_cube(const int el_id) const
{
	return
	elements_tag_[el_id] == ElementType::InterfaceCube ||

	elements_tag_[el_id] == ElementType::RegularInteriorCube ||
	elements_tag_[el_id] == ElementType::RegularBoundaryCube ||

	elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube ||
	elements_tag_[el_id] == ElementType::SimpleSingularBoundaryCube ||

	elements_tag_[el_id] == ElementType::MultiSingularInteriorCube ||
	elements_tag_[el_id] == ElementType::MultiSingularBoundaryCube;
}

// -----------------------------------------------------------------------------

bool poly_fem::Mesh::is_polytope(const int el_id) const
{
	return
	elements_tag_[el_id] == ElementType::InteriorPolytope ||
	elements_tag_[el_id] == ElementType::BoundaryPolytope;
}

bool poly_fem::Mesh::is_simplex(const int el_id) const
{
	return
	elements_tag_[el_id] == ElementType::Simplex;
}
