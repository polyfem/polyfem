////////////////////////////////////////////////////////////////////////////////
#include <polyfem/MeshesJSON.hpp>

#include <polyfem/Mesh.hpp>
#include <polyfem/CMesh2D.hpp>
#include <polyfem/NCMesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <polyfem/MeshUtils.hpp>
#include <polyfem/StringUtils.hpp>
#include <polyfem/MshReader.hpp>

#include <polyfem/Logger.hpp>
#include <polyfem/JSONUtils.hpp>

#include <Eigen/Core>
////////////////////////////////////////////////////////////////////////////////

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
	std::vector<int> body_vertices_start;
	std::vector<int> body_faces_start;
	std::vector<int> body_ids;
	std::vector<int> boundary_ids;
	std::vector<std::string> bc_tag_paths;

	body_faces_start.push_back(0);

	for (int i = 0; i < meshes.size(); i++)
	{
		json jmesh;
		apply_default_mesh_parameters(meshes[i], jmesh, fmt::format("/meshes[{}]", i));
		create_from_json(
			jmesh, root_path, vertices, cells, elements, weights,
			body_vertices_start, body_faces_start, body_ids, boundary_ids,
			bc_tag_paths);
	}

	////////////////////////////////////////////////////////////////////////////

	if (vertices.size() == 0)
		return nullptr;

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

	////////////////////////////////////////////////////////////////////////////

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

	////////////////////////////////////////////////////////////////////////////

	mesh->set_body_ids(body_ids);

	////////////////////////////////////////////////////////////////////////////

	mesh->set_multibody_boundary_ids(
		body_vertices_start, body_faces_start, boundary_ids, bc_tag_paths);

	////////////////////////////////////////////////////////////////////////////

	return mesh;
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::apply_default_mesh_parameters(const json &mesh_in, json &mesh_out, const std::string &path_prefix)
{
	// NOTE: All units by default are expressed in standard SI units
	// • position: position of the model origin
	// • rotation: degrees as XYZ euler angles around the model origin
	// • scale: scale the vertices around the model origin
	// • dimensions: dimensions of the scaled object (mutually exclusive to
	//               "scale")
	// • enabled: skip the body if this field is false
	// NOTE: default body id is 0 loaded from fem mesh
	mesh_out = R"({
				"mesh": null,
				"position": [0.0, 0.0, 0.0],
				"rotation": [0.0, 0.0, 0.0],
				"rotation_mode": "xyz",
				"scale": [1.0, 1.0, 1.0],
				"dimensions": null,
				"enabled": true,
				"body_id": null,
				"boundary_id": 0,
				"bc_tag": "",
				"displacement": [0.0, 0.0, 0.0]
			})"_json;
	check_for_unknown_args(mesh_out, mesh_in, path_prefix);
	mesh_out.merge_patch(mesh_in);
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::create_from_json(
	const json &jmesh,
	const std::string &root_path,
	Eigen::MatrixXd &vertices,
	Eigen::MatrixXi &cells,
	std::vector<std::vector<int>> &elements,
	std::vector<std::vector<double>> &weights,
	std::vector<int> &body_vertices_start,
	std::vector<int> &body_faces_start,
	std::vector<int> &body_ids,
	std::vector<int> &boundary_ids,
	std::vector<std::string> &bc_tag_paths)
{
	using namespace polyfem;

	if (!jmesh["enabled"].get<bool>())
		return;

	if (!is_param_valid(jmesh, "mesh"))
	{
		logger().error("Mesh {} is mising a \"mesh\" field", jmesh);
		return;
	}

	const std::string mesh_path = resolve_path(jmesh["mesh"], root_path);

	Eigen::MatrixXd tmp_vertices;
	Eigen::MatrixXi tmp_cells;
	std::vector<std::vector<int>> tmp_elements;
	std::vector<std::vector<double>> tmp_weights;
	Eigen::VectorXi tmp_body_ids;
	read_fem_mesh(mesh_path, tmp_vertices, tmp_cells, tmp_elements, tmp_weights, tmp_body_ids);

	if (tmp_vertices.size() == 0 || tmp_cells.size() == 0)
		return;

	if (vertices.cols() != 0 && vertices.cols() != tmp_vertices.cols())
	{
		logger().error("Mixed dimension meshes is not implemented!");
		return;
	}

	if (cells.size() != 0 && cells.cols() != tmp_cells.cols())
	{
		logger().error("Mixed tet and hex (tri and quad) meshes is not implemented!");
		return;
	}

	////////////////////////////////////////////////////////////////////////////

	MatrixNd affine_transform;
	RowVectorNd translation;
	mesh_transform_from_json(jmesh, mesh_dimensions(tmp_vertices), affine_transform, translation);
	transform_mesh(affine_transform, translation, tmp_vertices);

	////////////////////////////////////////////////////////////////////////////

	body_vertices_start.push_back(vertices.rows());
	const int dim = std::max(vertices.cols(), tmp_vertices.cols());
	assert(dim == 2 || dim == 3);
	vertices.conservativeResize(vertices.rows() + tmp_vertices.rows(), dim);
	vertices.bottomRows(tmp_vertices.rows()) = tmp_vertices;

	////////////////////////////////////////////////////////////////////////////

	const int cell_cols = std::max(cells.cols(), tmp_cells.cols());
	cells.conservativeResize(cells.rows() + tmp_cells.rows(), cell_cols);
	cells.bottomRows(tmp_cells.rows()) = tmp_cells.array() + body_vertices_start.back(); // Shift vertex ids in cells

	////////////////////////////////////////////////////////////////////////////

	// Shift vertex ids in elements
	for (auto &element : tmp_elements)
	{
		for (auto &id : element)
		{
			id += body_vertices_start.back();
		}
	}
	elements.insert(elements.end(), tmp_elements.begin(), tmp_elements.end());

	////////////////////////////////////////////////////////////////////////////

	weights.insert(weights.end(), tmp_weights.begin(), tmp_weights.end());

	////////////////////////////////////////////////////////////////////////////

	body_faces_start.push_back(body_faces_start.back() + count_faces(dim, tmp_cells));

	////////////////////////////////////////////////////////////////////////////
	if (!is_param_valid(jmesh, "body_id")) // Constant body id has priority over mesh's stored ids
	{
		tmp_body_ids.setConstant(tmp_cells.rows(), jmesh["body_id"].get<int>());
	}
	body_ids.insert(body_ids.end(), tmp_body_ids.data(), tmp_body_ids.data() + tmp_body_ids.size());

	////////////////////////////////////////////////////////////////////////////

	boundary_ids.push_back(jmesh["boundary_id"].get<int>());

	////////////////////////////////////////////////////////////////////////////

	bc_tag_paths.push_back(resolve_path(jmesh["bc_tag"], root_path));
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::Mesh::set_multibody_boundary_ids(
	const std::vector<int> &body_vertices_start,
	const std::vector<int> &body_faces_start,
	const std::vector<int> &boundary_ids,
	const std::vector<std::string> &bc_tag_paths)
{
	// TODO: Make sure this preserve default boundary_ids
	for (auto &id : boundary_ids_)
		id = -1;

	assert(body_vertices_start.size() == boundary_ids.size());
	compute_boundary_ids([&](const std::vector<int> &vis, bool is_boundary) {
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

	assert(boundary_ids_.size() == (is_volume() ? n_faces() : n_edges()));
	assert(body_faces_start.back() == boundary_ids_.size());
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
			assert(bindex < boundary_ids_.size());
			std::istringstream(line) >> boundary_ids_[bindex];
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
}

////////////////////////////////////////////////////////////////////////////////

polyfem::VectorNd polyfem::mesh_dimensions(const Eigen::MatrixXd &vertices)
{
	return (vertices.colwise().maxCoeff() - vertices.colwise().minCoeff()).cwiseAbs();
}

void polyfem::mesh_transform_from_json(
	const json &mesh,
	const VectorNd &initial_dimensions,
	MatrixNd &affine_transform,
	RowVectorNd &translation)
{
	const int dim = initial_dimensions.size();
	assert(dim == 2 || dim == 3);

	RowVectorNd scale;
	if (mesh["dimensions"].is_array()) // default is nullptr
	{
		scale = mesh["dimensions"];
		const int scale_size = scale.size();
		scale.conservativeResize(dim);
		if (scale_size < dim)
			scale.tail(dim - scale_size).setZero();
		scale.array() /=
			(initial_dimensions.array() == 0).select(1.0, initial_dimensions).array();
	}
	else if (mesh["scale"].is_number())
	{
		scale.setConstant(dim, mesh["scale"].get<double>());
	}
	else
	{
		assert(mesh["scale"].is_array());
		scale = mesh["scale"];
		const int scale_size = scale.size();
		scale.conservativeResize(dim);
		if (scale_size < dim)
			scale.tail(dim - scale_size).setZero();
	}

	// Rotate around the models origin NOT the bodies center of mass.
	// We could expose this choice as a "rotate_around" field.
	MatrixNd R = MatrixNd::Identity(dim, dim);
	if (dim == 2 && mesh["rotation"].is_number())
	{
		R = Eigen::Rotation2Dd(
				deg2rad(mesh["rotation"].get<double>()))
				.toRotationMatrix();
	}
	else if (dim == 3)
	{
		R = to_rotation_matrix(mesh["rotation"], mesh["rotation_mode"]);
	}

	RowVectorNd position = mesh["position"];
	const int position_size = position.size();
	position.conservativeResize(dim);
	if (position_size < dim)
		position.tail(dim - position_size).setZero();

	affine_transform = R * scale.asDiagonal();
	translation = position;
}

void polyfem::transform_mesh(const MatrixNd &affine_transform, const RowVectorNd &translation, Eigen::MatrixXd &vertices)
{
	vertices *= affine_transform.transpose(); // (T*Vᵀ)ᵀ = V*Tᵀ
	vertices.rowwise() += translation;
}

void polyfem::transform_mesh_from_json(const json &mesh, Eigen::MatrixXd &vertices)
{
	MatrixNd affine_transform;
	RowVectorNd translation;
	mesh_transform_from_json(mesh, mesh_dimensions(vertices), affine_transform, translation);
	transform_mesh(affine_transform, translation, vertices);
}