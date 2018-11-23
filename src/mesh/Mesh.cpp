////////////////////////////////////////////////////////////////////////////////
#include <polyfem/Mesh.hpp>
#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <polyfem/StringUtils.hpp>
#include <polyfem/MshReader.hpp>

#include <polyfem/Logger.hpp>

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

std::unique_ptr<polyfem::Mesh> polyfem::Mesh::create(GEO::Mesh &meshin) {
	if (is_planar(meshin)) {
		std::unique_ptr<polyfem::Mesh> mesh = std::make_unique<Mesh2D>();
		if (mesh->load(meshin)) {
			return mesh;
		}
	} else {
		std::unique_ptr<polyfem::Mesh> mesh = std::make_unique<Mesh3D>();
		meshin.cells.connect();
		if (mesh->load(meshin)) {
			return mesh;
		}
	}

	logger().error("Failed to load mesh");
	return nullptr;
}

std::unique_ptr<polyfem::Mesh> polyfem::Mesh::create(const std::string &path, const bool force_linear_geometry) {
	std::string lowername = path;

	std::transform(lowername.begin(), lowername.end(), lowername.begin(), ::tolower);
	if (StringUtils::endswidth(lowername, ".hybrid")) {
		std::unique_ptr<polyfem::Mesh> mesh = std::make_unique<Mesh3D>();
		if (mesh->load(path)){
			return mesh;
		}
	}
	else if (StringUtils::endswidth(lowername, ".msh"))
	{
		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells;
		std::vector<std::vector<int>> elements;

		if(!MshReader::load(path, vertices, cells, elements))
			return nullptr;

		std::unique_ptr<polyfem::Mesh> mesh;
		if(cells.cols() == 3)
			mesh = std::make_unique<Mesh2D>();
		else
			mesh = std::make_unique<Mesh3D>();

		mesh->build_from_matrices(vertices, cells);
		if(!force_linear_geometry)
			mesh->attach_higher_order_nodes(vertices, elements);
		return mesh;
	}
	else {
		GEO::Mesh tmp;
		if (GEO::mesh_load(path, tmp)) {
			return create(tmp);
		}
	}
	logger().error("Failed to load mesh: {}", path);
	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::Mesh::edge_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_edges(), dimension());
	for (int e = 0; e < n_edges(); ++e) {
		barycenters.row(e) = edge_barycenter(e);
	}
}

void polyfem::Mesh::face_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_faces(), dimension());
	for (int f = 0; f < n_faces(); ++f) {
		barycenters.row(f) = face_barycenter(f);
	}
}

void polyfem::Mesh::cell_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_cells(), dimension());
	for (int c = 0; c < n_cells(); ++c) {
		barycenters.row(c) = cell_barycenter(c);
	}
}

////////////////////////////////////////////////////////////////////////////////

//Queries on the tags
bool polyfem::Mesh::is_spline_compatible(const int el_id) const
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

bool polyfem::Mesh::is_cube(const int el_id) const
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

bool polyfem::Mesh::is_polytope(const int el_id) const
{
	return
	elements_tag_[el_id] == ElementType::InteriorPolytope ||
	elements_tag_[el_id] == ElementType::BoundaryPolytope;
}

void polyfem::Mesh::load_boundary_ids(const std::string &path)
{
	boundary_ids_.resize(is_volume()? n_faces() : n_edges());

	std::ifstream file(path);

	std::string line;
	int bindex = 0;
	while (std::getline(file, line))
	{
		std::istringstream iss(line);
		int v;
		iss >> v;
		boundary_ids_[bindex] = v;

		++bindex;
	}

	assert(boundary_ids_.size() == size_t(bindex));

	file.close();
}

bool polyfem::Mesh::is_simplex(const int el_id) const
{
	return
	elements_tag_[el_id] == ElementType::Simplex;
}
