////////////////////////////////////////////////////////////////////////////////
#include <polyfem/Mesh.hpp>
#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>

#include <polyfem/StringUtils.hpp>
#include <polyfem/MshReader.hpp>

#include <polyfem/Logger.hpp>

#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_geometry.h>

#include <ghc/fs_std.hpp> // filesystem

#include <Eigen/Geometry>

#include <igl/boundary_facets.h>
#include <igl/PI.h>
////////////////////////////////////////////////////////////////////////////////

namespace
{

	bool is_planar(const GEO::Mesh &M)
	{
		if (M.vertices.dimension() == 2)
		{
			return true;
		}
		assert(M.vertices.dimension() == 3);
		GEO::vec3 min_corner, max_corner;
		GEO::get_bbox(M, &min_corner[0], &max_corner[0]);
		return (max_corner[2] - min_corner[2]) < 1e-5;
	}

} // anonymous namespace

std::unique_ptr<polyfem::Mesh> polyfem::Mesh::create(GEO::Mesh &meshin)
{
	if (is_planar(meshin))
	{
		std::unique_ptr<polyfem::Mesh> mesh = std::make_unique<Mesh2D>();
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

std::unique_ptr<polyfem::Mesh> polyfem::Mesh::create(const std::string &path)
{
	if (!fs::exists(path))
	{
		logger().error(path.empty() ? "No mesh provided!" : "Mesh file does not exist: {}", path);
		return nullptr;
	}

	std::string lowername = path;

	std::transform(lowername.begin(), lowername.end(), lowername.begin(), ::tolower);
	if (StringUtils::endswidth(lowername, ".hybrid"))
	{
		std::unique_ptr<polyfem::Mesh> mesh = std::make_unique<Mesh3D>();
		if (mesh->load(path))
		{
			return mesh;
		}
	}
	else if (StringUtils::endswidth(lowername, ".msh"))
	{
		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells;
		std::vector<std::vector<int>> elements;
		std::vector<std::vector<double>> weights;

		if (!MshReader::load(path, vertices, cells, elements, weights))
			return nullptr;

		std::unique_ptr<polyfem::Mesh> mesh;
		if (vertices.cols() == 2)
			mesh = std::make_unique<Mesh2D>();
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

		return mesh;
	}
	else
	{
		GEO::Mesh tmp;
		if (GEO::mesh_load(path, tmp))
		{
			return create(tmp);
		}
	}
	logger().error("Failed to load mesh: {}", path);
	return nullptr;
}

template <typename Derived>
void from_json(const json &j, Eigen::MatrixBase<Derived> &v)
{
	auto jv = j.get<std::vector<typename Derived::Scalar>>();
	v = Eigen::Map<Derived>(jv.data(), long(jv.size()));
}

template <typename T>
inline T deg2rad(T deg)
{
	return deg / 180 * igl::PI;
}

Eigen::Matrix3d build_rotation_matrix(const Eigen::VectorXd &r, std::string mode = "xyz")
{
	std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

	if (mode == "axis_angle")
	{
		assert(r.size() == 4);
		double angle = r[0];
		Eigen::Vector3d axis = r.tail<3>().normalized();
		return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
	}

	if (mode == "quaternion")
	{
		assert(r.size() == 4);
		Eigen::Vector4d q = r.normalized();
		return Eigen::Quaterniond(q).toRotationMatrix();
	}

	if (mode == "rotation_vector")
	{
		assert(r.size() == 3);
		double angle = r.norm();
		if (angle != 0)
		{
			return Eigen::AngleAxisd(angle, r / angle).toRotationMatrix();
		}
		else
		{
			return Eigen::Matrix3d::Identity();
		}
	}

	Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

	assert(r.size() >= 3);
	for (int i = 0; i < mode.size(); i++)
	{
		int j = mode[i] - 'x';
		assert(j >= 0 && j < 3);
		Eigen::Vector3d axis = Eigen::Vector3d::Zero();
		axis[j] = 1;
		R = Eigen::AngleAxisd(r[j], axis).toRotationMatrix() * R;
	}

	return R;
}

std::unique_ptr<polyfem::Mesh> polyfem::Mesh::create(const std::vector<json> &meshes)
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
	std::vector<int> body_ids;
	std::vector<int> boundary_ids;
	int dim = 0;
	int cell_cols = 0;

	for (int i = 0; i < meshes.size(); i++)
	{
		// NOTE: All units by default are expressed in standard SI units
		// • position: position of the model origin
		// • rotation: degrees as XYZ euler angles around the model origin
		// • scale: scale the vertices around the model origin
		// • dimensions: dimensions of the scaled object (mutually exclusive to
		//               "scale")
		// • enabled: skip the body if this field is false
		json jmesh = R"({
				"position": [0.0, 0.0, 0.0],
				"rotation": [0.0, 0.0, 0.0],
				"rotation_mode": "xyz",
				"scale": [1.0, 1.0, 1.0],
				"enabled": true,
				"body_id": 0,
				"boundary_id": 0
			})"_json;
		jmesh.merge_patch(meshes[i]);

		if (!jmesh["enabled"].get<bool>())
		{
			continue;
		}

		if (!jmesh.contains("mesh"))
		{
			logger().error("Mesh {:d} is mising a \"mesh\" field", i);
			continue;
		}

		std::string mesh_path = jmesh["mesh"];
		std::string lowername = mesh_path;
		std::transform(
			lowername.begin(), lowername.end(), lowername.begin(), ::tolower);
		if (!StringUtils::endswidth(lowername, ".msh"))
		{
			logger().error("Unsupported mesh type in meshes: {}", mesh_path);
			continue;
		}

		Eigen::MatrixXd tmp_vertices;
		Eigen::MatrixXi tmp_cells;
		std::vector<std::vector<int>> tmp_elements;
		std::vector<std::vector<double>> tmp_weights;

		if (!MshReader::load(
				mesh_path, tmp_vertices, tmp_cells, tmp_elements, tmp_weights))
		{
			logger().error("Unable to load mesh: {}", mesh_path);
			continue;
		}

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
			logger().error("Mixed tet and hex meshes is not implemented!");
			continue;
		}

		RowVectorNd scale;
		if (jmesh.contains("dimensions"))
		{
			VectorNd initial_dimensions =
				(vertices.colwise().maxCoeff() - vertices.colwise().minCoeff())
					.cwiseAbs();
			initial_dimensions =
				(initial_dimensions.array() == 0).select(1, initial_dimensions);
			from_json(jmesh["dimensions"], scale);
			assert(scale.size() >= dim);
			scale.conservativeResize(dim);
			scale.array() /= initial_dimensions.array();
		}
		else if (jmesh["scale"].is_number())
		{
			scale.setConstant(dim, jmesh["scale"].get<double>());
		}
		else
		{
			assert(jmesh["scale"].is_array());
			from_json(jmesh["scale"], scale);
			assert(scale.size() >= dim);
			scale.conservativeResize(dim);
		}
		tmp_vertices *= scale.asDiagonal();

		// Rotate around the models origin NOT the bodies center of mass.
		// We could expose this choice as a "rotate_around" field.
		MatrixNd R;
		if (jmesh["rotation"].is_number())
		{
			assert(dim == 2);
			R = Eigen::Rotation2Dd(
					deg2rad(jmesh["rotation"].get<double>()))
					.toRotationMatrix();
		}
		else
		{
			assert(dim == 3);
			assert(jmesh["rotation"].is_array());
			Eigen::Vector3d rot;
			from_json(jmesh["rotation"], rot);
			rot = deg2rad(rot);
			// XYZ Euler angles, this is arbitrary and based on the default
			// in Blender. An alternative is to provide a field
			// "rotation_type" which specifies the type of rotation encoded
			// in `rot`.
			R = build_rotation_matrix(rot, jmesh["rotation_mode"].get<std::string>());
		}
		tmp_vertices *= R.transpose(); // (R*Vᵀ)ᵀ = V*Rᵀ

		RowVectorNd position;
		from_json(jmesh["position"], position);
		assert(position.size() >= dim);
		position.conservativeResize(dim);
		tmp_vertices.rowwise() += position;

		size_t vertices_offset = vertices.rows();
		vertices.conservativeResize(
			vertices.rows() + tmp_vertices.rows(), dim);
		vertices.bottomRows(tmp_vertices.rows()) = tmp_vertices;

		cells.conservativeResize(cells.rows() + tmp_cells.rows(), cell_cols);
		cells.bottomRows(tmp_cells.rows()) = tmp_cells.array() + vertices_offset;

		for (auto &element : tmp_elements)
		{
			for (auto &id : element)
			{
				id += vertices_offset;
			}
		}
		elements.insert(elements.end(), tmp_elements.begin(), tmp_elements.end());

		weights.insert(weights.end(), tmp_weights.begin(), tmp_weights.end());

		for (int ci = 0; ci < tmp_cells.rows(); ci++)
		{
			body_ids.push_back(jmesh["body_id"].get<int>());
		}

		if (cell_cols == 3 || cell_cols == 4)
		{
			// NOTE: This will make every face in the mesh a boundary
			Eigen::MatrixXi BF;
			igl::boundary_facets(tmp_cells, BF);
			// Every face that is not a boundary will have exactly two copies
			assert(BF.rows() % 2 == 0);
			int num_faces = (4 * tmp_cells.rows() - BF.rows()) / 2 + BF.rows();
			for (int fi = 0; fi < num_faces; fi++)
			{
				boundary_ids.push_back(jmesh["boundary_id"].get<int>());
			}
		}
		else if (jmesh["boundary_id"].get<int>() > 0)
		{
			logger().error("mesh `boundary_id` only implemented for triangle or tet meshes!");
		}
	}

	if (vertices.size() == 0)
	{
		return nullptr;
	}

	std::unique_ptr<polyfem::Mesh> mesh;
	if (vertices.cols() == 2)
	{
		mesh = std::make_unique<Mesh2D>();
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
	mesh->set_boundary_ids(boundary_ids);

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
