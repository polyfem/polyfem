#include "MeshLoader.hpp"

#include <polyfem/mesh/MeshReader.hpp>
#include <polyfem/utils/Logger.hpp>

#include <algorithm>

namespace polyfem::mesh
{
	namespace
	{
		std::string child_path(const std::string &group, const std::string &name)
		{
			return (std::filesystem::path(group) / name).lexically_normal().generic_string();
		}

		std::vector<std::vector<int>> unpack_connectivity(
			const std::vector<int> &values,
			const std::vector<long> &offsets,
			const io::ResourceIO &resources,
			const std::string &path)
		{
			if (offsets.empty() || offsets.front() != 0 || offsets.back() != long(values.size()))
				log_and_throw_error("Invalid ragged offsets in {}.", resources.describe(path));
			std::vector<std::vector<int>> result(offsets.size() - 1);
			for (size_t i = 0; i + 1 < offsets.size(); ++i)
			{
				if (offsets[i] > offsets[i + 1])
					log_and_throw_error("Offsets in {} are not monotone.", resources.describe(path));
				result[i].assign(values.begin() + offsets[i], values.begin() + offsets[i + 1]);
			}
			return result;
		}

		std::vector<std::vector<double>> unpack_weights(
			const std::vector<double> &values,
			const std::vector<long> &offsets,
			const io::ResourceIO &resources,
			const std::string &path)
		{
			if (offsets.empty() || offsets.front() != 0 || offsets.back() != long(values.size()))
				log_and_throw_error("Invalid weight offsets in {}.", resources.describe(path));
			std::vector<std::vector<double>> result(offsets.size() - 1);
			for (size_t i = 0; i + 1 < offsets.size(); ++i)
				result[i].assign(values.begin() + offsets[i], values.begin() + offsets[i + 1]);
			return result;
		}
	} // namespace

	void MeshLoader::validate_group(const std::string &group, const std::string &expected_type) const
	{
		if (resources_.exists(child_path(group, "v")) || resources_.exists(child_path(group, "c")))
			log_and_throw_error(
				"Historical mesh datasets 'v'/'c' are unsupported in {}; use 'vertices'/'cells'.",
				resources_.describe(group));
		if (!resources_.has_attribute(group, "schema_version"))
			log_and_throw_error("Mesh group {} is missing schema_version metadata.", resources_.describe(group));
		const long version = resources_.read_integer_attribute(group, "schema_version");
		if (version != MESH_SCHEMA_VERSION)
			log_and_throw_error(
				"Unsupported mesh schema version {} in {}; expected {}.",
				version, resources_.describe(group), MESH_SCHEMA_VERSION);
		if (!resources_.has_attribute(group, "mesh_type") || resources_.read_string_attribute(group, "mesh_type") != expected_type)
			log_and_throw_error(
				"Mesh group {} does not declare mesh_type='{}'.",
				resources_.describe(group), expected_type);
		if (!resources_.has_attribute(group, "dimension"))
			log_and_throw_error("Mesh group {} is missing dimension metadata.", resources_.describe(group));
	}

	std::unique_ptr<Mesh> MeshLoader::load_fem(const std::string &path, const bool non_conforming) const
	{
		if (!resources_.exists(path))
			log_and_throw_error("Mesh resource {} does not exist.", resources_.describe(path));

		MeshData data{Eigen::MatrixXd(), Eigen::MatrixXi()};
		if (!resources_.is_group(path))
		{
			std::string extension = std::filesystem::path(path).extension().string();
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
			if (extension == ".msh")
				data = MeshReader::read_msh(resources_.materialize(path));
			else if (extension == ".hybrid")
			{
				auto input = resources_.open(path, false);
				data = MeshReader::read_hybrid(*input, resources_.describe(path));
			}
			else
				data = MeshReader::read_geogram(resources_.materialize(path));
			return Mesh::create(std::move(data), non_conforming);
		}

		validate_group(path, "fem");
		const std::string vertices_path = child_path(path, "vertices");
		const std::string cells_path = child_path(path, "cells");
		if (!resources_.exists(vertices_path) || !resources_.exists(cells_path))
			log_and_throw_error("FEM mesh group {} requires vertices and cells.", resources_.describe(path));
		Eigen::MatrixXd vertices = resources_.read_matrix(vertices_path);
		Eigen::MatrixXi cells = resources_.read_int_matrix(cells_path);
		const long dimension = resources_.read_integer_attribute(path, "dimension");
		if ((dimension != 2 && dimension != 3) || vertices.cols() != dimension)
			log_and_throw_error("Invalid dimension metadata in {}.", resources_.describe(path));
		if (vertices.rows() == 0 || cells.rows() == 0 || cells.cols() < 3)
			log_and_throw_error("Empty or invalid connectivity in {}.", resources_.describe(path));
		if (cells.minCoeff() < -1 || cells.maxCoeff() >= vertices.rows())
			log_and_throw_error("Connectivity in {} references an invalid vertex.", resources_.describe(path));

		data = MeshData(std::move(vertices), std::move(cells));

		const std::string body_ids_path = child_path(path, "body_ids");
		if (resources_.exists(body_ids_path))
			data.body_ids = resources_.read_int_vector(body_ids_path);
		const std::string geometry_ids_path = child_path(path, "geometry_ids");
		if (resources_.exists(geometry_ids_path))
			data.geometry_ids = resources_.read_int_vector(geometry_ids_path);
		const std::string boundary_ids_path = child_path(path, "boundary_ids");
		if (resources_.exists(boundary_ids_path))
		{
			const std::string boundary_elements_path = child_path(path, "boundary_elements");
			if (!resources_.exists(boundary_elements_path))
				log_and_throw_error("{} requires boundary_elements.", resources_.describe(boundary_ids_path));
			const Eigen::MatrixXi elements = resources_.read_int_matrix(boundary_elements_path);
			data.boundary_ids = resources_.read_int_vector(boundary_ids_path);
			data.boundary_elements.resize(elements.rows());
			for (int i = 0; i < elements.rows(); ++i)
			{
				for (int j = 0; j < elements.cols() && elements(i, j) >= 0; ++j)
					data.boundary_elements[i].push_back(elements(i, j));
			}
		}

		const std::string higher_nodes_path = child_path(path, "higher_order_nodes");
		const std::string higher_connectivity_path = child_path(path, "higher_order_connectivity");
		const std::string higher_offsets_path = child_path(path, "higher_order_offsets");
		const bool any_higher = resources_.exists(higher_nodes_path)
								|| resources_.exists(higher_connectivity_path) || resources_.exists(higher_offsets_path);
		if (any_higher)
		{
			if (!resources_.exists(higher_nodes_path) || !resources_.exists(higher_connectivity_path) || !resources_.exists(higher_offsets_path))
				log_and_throw_error("Higher-order data in {} is incomplete.", resources_.describe(path));
			data.higher_order_connectivity = unpack_connectivity(
				resources_.read_int_vector(higher_connectivity_path), resources_.read_long_vector(higher_offsets_path),
				resources_, higher_offsets_path);
			data.higher_order_nodes = resources_.read_matrix(higher_nodes_path);
		}

		const std::string weights_path = child_path(path, "higher_order_weights");
		if (resources_.exists(weights_path))
		{
			const std::string offsets_path = child_path(path, "higher_order_weight_offsets");
			if (!resources_.exists(offsets_path))
				log_and_throw_error("{} requires higher_order_weight_offsets.", resources_.describe(weights_path));
			data.higher_order_weights = unpack_weights(
				resources_.read_double_vector(weights_path), resources_.read_long_vector(offsets_path),
				resources_, offsets_path);
		}

		const std::string faces_path = child_path(path, "faces");
		const std::string face_offsets_path = child_path(path, "face_offsets");
		const std::string cell_faces_path = child_path(path, "cell_faces");
		const std::string cell_face_offsets_path = child_path(path, "cell_face_offsets");
		const bool any_polyhedral = resources_.exists(faces_path) || resources_.exists(face_offsets_path)
									|| resources_.exists(cell_faces_path) || resources_.exists(cell_face_offsets_path)
									|| resources_.exists(child_path(path, "cell_face_orientations"))
									|| resources_.exists(child_path(path, "cell_is_hex"))
									|| resources_.exists(child_path(path, "cell_kernel_points"));
		if (any_polyhedral)
		{
			for (const std::string &required : {
					 faces_path, face_offsets_path, cell_faces_path, cell_face_offsets_path,
					 child_path(path, "cell_face_orientations"), child_path(path, "cell_is_hex"),
					 child_path(path, "cell_kernel_points")})
				if (!resources_.exists(required))
					log_and_throw_error("Polyhedral mesh group {} is missing {}.", resources_.describe(path), required);
			data.faces = unpack_connectivity(
				resources_.read_int_vector(faces_path), resources_.read_long_vector(face_offsets_path),
				resources_, face_offsets_path);
			const std::vector<long> cell_face_offsets = resources_.read_long_vector(cell_face_offsets_path);
			data.cell_faces = unpack_connectivity(
				resources_.read_int_vector(cell_faces_path), cell_face_offsets,
				resources_, cell_face_offsets_path);
			data.cell_face_orientations = unpack_connectivity(
				resources_.read_int_vector(child_path(path, "cell_face_orientations")), cell_face_offsets,
				resources_, cell_face_offsets_path);
			const std::vector<int> cell_is_hex = resources_.read_int_vector(child_path(path, "cell_is_hex"));
			data.cell_is_hex.assign(cell_is_hex.begin(), cell_is_hex.end());
			data.cell_kernel_points = resources_.read_matrix(child_path(path, "cell_kernel_points"));
		}
		return Mesh::create(std::move(data), non_conforming);
	}

	SurfaceMesh MeshLoader::load_surface(const std::string &path) const
	{
		if (!resources_.exists(path))
			log_and_throw_error("Surface resource {} does not exist.", resources_.describe(path));
		SurfaceMesh result;
		if (!resources_.is_group(path))
		{
			const std::filesystem::path materialized = resources_.materialize(path);
			std::string extension = std::filesystem::path(path).extension().string();
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
			if (extension == ".msh")
				result = MeshReader::read_msh_surface(materialized);
			else if (extension == ".obj")
				result = MeshReader::read_obj_surface(materialized);
			else if (!MeshReader::read_triangle_surface(materialized, result))
				result = MeshReader::read_geogram_surface(materialized);
			return result;
		}

		validate_group(path, "surface");
		const std::string vertices_path = child_path(path, "vertices");
		if (!resources_.exists(vertices_path))
			log_and_throw_error("Surface mesh group {} requires vertices.", resources_.describe(path));
		result.vertices = resources_.read_matrix(vertices_path);
		if (result.vertices.cols() != resources_.read_integer_attribute(path, "dimension"))
			log_and_throw_error("Invalid surface dimension metadata in {}.", resources_.describe(path));
		const std::string points_path = child_path(path, "points");
		if (resources_.exists(points_path))
		{
			const std::vector<int> values = resources_.read_int_vector(points_path);
			result.points = Eigen::Map<const Eigen::VectorXi>(values.data(), values.size());
		}
		const std::string edges_path = child_path(path, "edges");
		if (resources_.exists(edges_path))
			result.edges = resources_.read_int_matrix(edges_path);
		const std::string faces_path = child_path(path, "faces");
		if (resources_.exists(faces_path))
			result.faces = resources_.read_int_matrix(faces_path);
		const auto valid = [&](const Eigen::MatrixXi &elements) {
			return elements.size() == 0 || (elements.minCoeff() >= 0 && elements.maxCoeff() < result.vertices.rows());
		};
		if ((result.points.size() && (result.points.minCoeff() < 0 || result.points.maxCoeff() >= result.vertices.rows()))
			|| !valid(result.edges) || !valid(result.faces))
			log_and_throw_error("Surface connectivity in {} references an invalid vertex.", resources_.describe(path));
		return result;
	}
} // namespace polyfem::mesh
