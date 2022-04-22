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

std::unique_ptr<polyfem::Mesh> polyfem::Mesh::create(const std::vector<json> &meshes, const std::string &root_path, const bool non_conforming)
{
	if (meshes.empty())
	{
		logger().error("Provided meshes is empty!");
		return nullptr;
	}

	Eigen::MatrixXd vertices;
	Eigen::MatrixXi cells;
	std::vector<std::vector<int>> elements;
	std::vector<std::vector<double>> weights;
	std::vector<int> body_vertices_start, body_faces_start;
	std::vector<int> body_ids, boundary_ids;

	std::vector<std::string> bc_tag_paths;

	int dim = 0;
	int cell_cols = 0;

	body_faces_start.push_back(0);

	for (int i = 0; i < meshes.size(); i++)
	{
		json jmesh;
		apply_default_mesh_parameters(meshes[i], jmesh, fmt::format("/meshes[{}]", i));

		if (!jmesh["enabled"].get<bool>())
		{
			continue;
		}

		if (!meshes[i].contains("mesh"))
		{
			logger().error("Mesh {} is mising a \"mesh\" field", meshes[i].get<std::string>());
			continue;
		}
		const std::string mesh_path = resolve_path(jmesh["mesh"], root_path);

		Eigen::MatrixXd tmp_vertices;
		Eigen::MatrixXi tmp_cells;
		std::vector<std::vector<int>> tmp_elements;
		std::vector<std::vector<double>> tmp_weights;
		Eigen::VectorXi tmp_body_ids;
		read_fem_mesh(mesh_path, tmp_vertices, tmp_cells, tmp_elements, tmp_weights, tmp_body_ids);

		if (tmp_vertices.size() == 0 || tmp_cells.size() == 0)
		{
			continue;
		}

		transform_mesh_from_json(jmesh, tmp_vertices);

		if (dim == 0)
		{
			dim = tmp_vertices.cols();
		}
		else if (dim != tmp_vertices.cols())
		{
			logger().error("Mixed dimension meshes is not implemented!");
			continue;
		}

		if (cell_cols == 0)
		{
			cell_cols = tmp_cells.cols();
		}
		else if (cell_cols != tmp_cells.cols())
		{
			logger().error("Mixed tet and hex (tri and quad) meshes is not implemented!");
			continue;
		}

		body_vertices_start.push_back(vertices.rows());
		vertices.conservativeResize(
			vertices.rows() + tmp_vertices.rows(), dim);
		vertices.bottomRows(tmp_vertices.rows()) = tmp_vertices;

		cells.conservativeResize(cells.rows() + tmp_cells.rows(), cell_cols);
		cells.bottomRows(tmp_cells.rows()) = tmp_cells.array() + body_vertices_start.back();

		for (auto &element : tmp_elements)
		{
			for (auto &id : element)
			{
				id += body_vertices_start.back();
			}
		}
		elements.insert(elements.end(), tmp_elements.begin(), tmp_elements.end());

		weights.insert(weights.end(), tmp_weights.begin(), tmp_weights.end());

		body_faces_start.push_back(body_faces_start.back() + count_faces(dim, tmp_cells));

		if (meshes[i].contains("body_id")) // Constant body id has priority over mesh's stored ids
			tmp_body_ids.setConstant(tmp_cells.rows(), jmesh["body_id"].get<int>());
		body_ids.insert(body_ids.end(), tmp_body_ids.data(), tmp_body_ids.data() + tmp_body_ids.size());

		boundary_ids.push_back(jmesh["boundary_id"].get<int>());

		bc_tag_paths.push_back(resolve_path(jmesh["bc_tag"], root_path));
	}

	if (vertices.size() == 0)
	{
		return nullptr;
	}

	std::unique_ptr<polyfem::Mesh> mesh;
	if (vertices.cols() == 2)
	{
		if (non_conforming)
			mesh = std::make_unique<NCMesh2D>();
		else
			mesh = std::make_unique<CMesh2D>();
	}
	else
	{
		mesh = std::make_unique<Mesh3D>();
	}

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

	for (auto &id : mesh->boundary_ids_)
		id = -1;

	assert(body_vertices_start.size() == boundary_ids.size());
	mesh->compute_boundary_ids([&](const std::vector<int> &vis, bool is_boundary) {
		if (!is_boundary)
		{
			return -1;
		}

		for (int i = 0; i < body_vertices_start.size() - 1; i++)
		{
			if (body_vertices_start[i] <= vis[0] && vis[0] < body_vertices_start[i + 1])
			{
				return boundary_ids[i];
			}
		}
		return boundary_ids.back();
	});

	assert(mesh->boundary_ids_.size() == (mesh->is_volume() ? mesh->n_faces() : mesh->n_edges()));
	assert(body_faces_start.back() == mesh->boundary_ids_.size());
	for (int i = 0; i < bc_tag_paths.size(); i++)
	{
		const std::string &path = bc_tag_paths[i];
		if (path.empty())
			continue;

		std::ifstream file(path);
		if (!file.is_open())
		{
			logger().error("Unable to open bc_tag file \"{}\"!", path);
			continue;
		}
		std::string line;
		int bindex = body_faces_start[i];
		while (std::getline(file, line))
		{
			assert(bindex < mesh->boundary_ids_.size());
			std::istringstream(line) >> mesh->boundary_ids_[bindex];
			bindex++;
		}

		if (bindex != body_faces_start[i + 1])
		{
			logger().error(
				"/meshes[{}]/bc_tag file \"{}\" is missing {} tag(s)!",
				i, path, body_faces_start[i + 1] - bindex);
			assert(false);
		}
	}

	return mesh;
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
