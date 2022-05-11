////////////////////////////////////////////////////////////////////////////////
#include <polyfem/Mesh.hpp>
#include <polyfem/CMesh2D.hpp>
#include <polyfem/NCMesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <polyfem/MeshUtils.hpp>
#include <polyfem/StringUtils.hpp>
#include <polyfem/MshReader.hpp>

#include <polyfem/Logger.hpp>

#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_geometry.h>

#include <Eigen/Geometry>

#include <igl/boundary_facets.h>

#include <filesystem>
////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<polyfem::Mesh> polyfem::Mesh::create(GEO::Mesh &meshin, const bool non_conforming)
{
	if (is_planar(meshin))
	{
		std::unique_ptr<polyfem::Mesh> mesh;
		if (non_conforming)
			mesh = std::make_unique<NCMesh2D>();
		else
			mesh = std::make_unique<CMesh2D>();
		if (mesh->load(meshin))
		{
			return mesh;
		}
	}
	else
	{
		std::unique_ptr<polyfem::Mesh> mesh = std::make_unique<Mesh3D>();
		meshin.cells.connect();
		if (mesh->load(meshin))
		{
			return mesh;
		}
	}

	logger().error("Failed to load mesh");
	return nullptr;
}

std::unique_ptr<polyfem::Mesh> polyfem::Mesh::create(const std::string &path, const bool non_conforming)
{
	if (!std::filesystem::exists(path))
	{
		logger().error(path.empty() ? "No mesh provided!" : "Mesh file does not exist: {}", path);
		return nullptr;
	}

	std::string lowername = path;

	std::transform(lowername.begin(), lowername.end(), lowername.begin(), ::tolower);
	if (StringUtils::endswith(lowername, ".hybrid"))
	{
		std::unique_ptr<polyfem::Mesh> mesh = std::make_unique<Mesh3D>();
		if (mesh->load(path))
		{
			return mesh;
		}
	}
	else if (StringUtils::endswith(lowername, ".msh"))
	{
		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells;
		std::vector<std::vector<int>> elements;
		std::vector<std::vector<double>> weights;
		Eigen::VectorXi body_ids;

		if (!MshReader::load(path, vertices, cells, elements, weights, body_ids))
			return nullptr;

		std::unique_ptr<polyfem::Mesh> mesh;
		if (vertices.cols() == 2)
			if (non_conforming)
				mesh = std::make_unique<NCMesh2D>();
			else
				mesh = std::make_unique<CMesh2D>();
		else
			mesh = std::make_unique<Mesh3D>();

		mesh->build_from_matrices(vertices, cells);
		// Only tris and tets
		if ((vertices.cols() == 2 && cells.cols() == 3) || (vertices.cols() == 3 && cells.cols() == 4))
		{
			mesh->attach_higher_order_nodes(vertices, elements);
			mesh->cell_weights_ = weights;
		}

		for (const auto &w : weights)
		{
			if (!w.empty())
			{
				mesh->is_rational_ = true;
				break;
			}
		}

		mesh->set_body_ids(body_ids);

		return mesh;
	}
	else
	{
		GEO::Mesh tmp;
		if (GEO::mesh_load(path, tmp))
		{
			return create(tmp, non_conforming);
		}
	}
	logger().error("Failed to load mesh: {}", path);
	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::Mesh::edge_barycenters(Eigen::MatrixXd &barycenters) const
{
	barycenters.resize(n_edges(), dimension());
	for (int e = 0; e < n_edges(); ++e)
	{
		barycenters.row(e) = edge_barycenter(e);
	}
}

void polyfem::Mesh::face_barycenters(Eigen::MatrixXd &barycenters) const
{
	barycenters.resize(n_faces(), dimension());
	for (int f = 0; f < n_faces(); ++f)
	{
		barycenters.row(f) = face_barycenter(f);
	}
}

void polyfem::Mesh::cell_barycenters(Eigen::MatrixXd &barycenters) const
{
	barycenters.resize(n_cells(), dimension());
	for (int c = 0; c < n_cells(); ++c)
	{
		barycenters.row(c) = cell_barycenter(c);
	}
}

////////////////////////////////////////////////////////////////////////////////

// Queries on the tags
bool polyfem::Mesh::is_spline_compatible(const int el_id) const
{
	if (is_volume())
	{
		return elements_tag_[el_id] == ElementType::RegularInteriorCube
			   || elements_tag_[el_id] == ElementType::RegularBoundaryCube;
		// || elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube
		// || elements_tag_[el_id] == ElementType::SimpleSingularBoundaryCube;
	}
	else
	{
		return elements_tag_[el_id] == ElementType::RegularInteriorCube
			   || elements_tag_[el_id] == ElementType::RegularBoundaryCube;
		// || elements_tag_[el_id] == ElementType::InterfaceCube
		// || elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube;
	}
}

// -----------------------------------------------------------------------------

bool polyfem::Mesh::is_cube(const int el_id) const
{
	return elements_tag_[el_id] == ElementType::InterfaceCube
		   || elements_tag_[el_id] == ElementType::RegularInteriorCube
		   || elements_tag_[el_id] == ElementType::RegularBoundaryCube
		   || elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube
		   || elements_tag_[el_id] == ElementType::SimpleSingularBoundaryCube
		   || elements_tag_[el_id] == ElementType::MultiSingularInteriorCube
		   || elements_tag_[el_id] == ElementType::MultiSingularBoundaryCube;
}

// -----------------------------------------------------------------------------

bool polyfem::Mesh::is_polytope(const int el_id) const
{
	return elements_tag_[el_id] == ElementType::InteriorPolytope
		   || elements_tag_[el_id] == ElementType::BoundaryPolytope;
}

void polyfem::Mesh::load_boundary_ids(const std::string &path)
{
	boundary_ids_.resize(is_volume() ? n_faces() : n_edges());

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
	return elements_tag_[el_id] == ElementType::Simplex;
}

std::vector<std::pair<int, int>> polyfem::Mesh::edges() const
{
	std::vector<std::pair<int, int>> res;
	res.reserve(n_edges());

	for (int e_id = 0; e_id < n_edges(); ++e_id)
	{
		const int e0 = edge_vertex(e_id, 0);
		const int e1 = edge_vertex(e_id, 1);

		res.emplace_back(std::min(e0, e1), std::max(e0, e1));
	}

	return res;
}

std::vector<std::vector<int>> polyfem::Mesh::faces() const
{
	std::vector<std::vector<int>> res(n_faces());

	for (int f_id = 0; f_id < n_faces(); ++f_id)
	{
		auto &tmp = res[f_id];
		for (int lv_id = 0; lv_id < n_face_vertices(f_id); ++lv_id)
			tmp.push_back(face_vertex(f_id, lv_id));

		std::sort(tmp.begin(), tmp.end());
	}

	return res;
}

std::unordered_map<std::pair<int, int>, size_t, polyfem::HashPair> polyfem::Mesh::edges_to_ids() const
{
	std::unordered_map<std::pair<int, int>, size_t, polyfem::HashPair> res;
	res.reserve(n_edges());

	for (int e_id = 0; e_id < n_edges(); ++e_id)
	{
		const int e0 = edge_vertex(e_id, 0);
		const int e1 = edge_vertex(e_id, 1);

		res[std::pair<int, int>(std::min(e0, e1), std::max(e0, e1))] = e_id;
	}

	return res;
}

std::unordered_map<std::vector<int>, size_t, polyfem::HashVector> polyfem::Mesh::faces_to_ids() const
{
	std::unordered_map<std::vector<int>, size_t, polyfem::HashVector> res;
	res.reserve(n_faces());

	for (int f_id = 0; f_id < n_faces(); ++f_id)
	{
		std::vector<int> f;
		f.reserve(n_face_vertices(f_id));
		for (int lv_id = 0; lv_id < n_face_vertices(f_id); ++lv_id)
			f.push_back(face_vertex(f_id, lv_id));
		std::sort(f.begin(), f.end());

		res[f] = f_id;
	}

	return res;
}
