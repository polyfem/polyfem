#include "MeshLoader.hpp"

#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/HashUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <unordered_map>

namespace polyfem::mesh
{
	namespace
	{
		std::string child_path(const std::string &group, const std::string &name)
		{
			return (std::filesystem::path(group) / name).lexically_normal().generic_string();
		}

		template <typename T>
		void require_size(
			const std::vector<T> &values,
			const size_t size,
			const io::ResourceIO &resources,
			const std::string &path)
		{
			if (values.size() != size)
				log_and_throw_error(
					"Dataset {} has {} entries; expected {}.",
					resources.describe(path), values.size(), size);
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
		if (!resources_.is_group(path))
		{
			auto mesh = Mesh::create(resources_.materialize(path).string(), non_conforming);
			if (!mesh)
				log_and_throw_error("Unable to decode FEM mesh {}.", resources_.describe(path));
			return mesh;
		}

		validate_group(path, "fem");
		const std::string vertices_path = child_path(path, "vertices");
		const std::string cells_path = child_path(path, "cells");
		if (!resources_.exists(vertices_path) || !resources_.exists(cells_path))
			log_and_throw_error("FEM mesh group {} requires vertices and cells.", resources_.describe(path));
		const Eigen::MatrixXd vertices = resources_.read_matrix(vertices_path);
		const Eigen::MatrixXi cells = resources_.read_int_matrix(cells_path);
		const long dimension = resources_.read_integer_attribute(path, "dimension");
		if ((dimension != 2 && dimension != 3) || vertices.cols() != dimension)
			log_and_throw_error("Invalid dimension metadata in {}.", resources_.describe(path));
		if (vertices.rows() == 0 || cells.rows() == 0 || cells.cols() < 3)
			log_and_throw_error("Empty or invalid connectivity in {}.", resources_.describe(path));
		if (cells.minCoeff() < 0 || cells.maxCoeff() >= vertices.rows())
			log_and_throw_error("Connectivity in {} references an invalid vertex.", resources_.describe(path));

		std::unique_ptr<Mesh> mesh = Mesh::create(vertices, cells, non_conforming);
		if (!mesh)
			log_and_throw_error("Unable to construct FEM mesh {}.", resources_.describe(path));

		const std::string body_ids_path = child_path(path, "body_ids");
		if (resources_.exists(body_ids_path))
		{
			const std::vector<int> values = resources_.read_int_vector(body_ids_path);
			require_size(values, mesh->n_elements(), resources_, body_ids_path);
			mesh->set_body_ids(values);
		}
		const std::string geometry_ids_path = child_path(path, "geometry_ids");
		if (resources_.exists(geometry_ids_path))
		{
			const std::vector<int> values = resources_.read_int_vector(geometry_ids_path);
			require_size(values, mesh->n_elements(), resources_, geometry_ids_path);
			mesh->set_geometry_ids(values);
		}
		const std::string boundary_ids_path = child_path(path, "boundary_ids");
		if (resources_.exists(boundary_ids_path))
		{
			const std::string boundary_elements_path = child_path(path, "boundary_elements");
			if (!resources_.exists(boundary_elements_path))
				log_and_throw_error("{} requires boundary_elements.", resources_.describe(boundary_ids_path));
			const Eigen::MatrixXi elements = resources_.read_int_matrix(boundary_elements_path);
			const std::vector<int> ids = resources_.read_int_vector(boundary_ids_path);
			require_size(ids, elements.rows(), resources_, boundary_ids_path);
			std::unordered_map<std::vector<int>, int, utils::HashVector> labels;
			for (int i = 0; i < elements.rows(); ++i)
			{
				std::vector<int> side(elements.row(i).data(), elements.row(i).data() + elements.cols());
				std::sort(side.begin(), side.end());
				labels.emplace(std::move(side), ids[i]);
			}
			mesh->compute_boundary_ids([&](const size_t primitive, const std::vector<int> &vertices, const RowVectorNd &, const bool) {
				std::vector<int> side = vertices;
				std::sort(side.begin(), side.end());
				const auto it = labels.find(side);
				return it == labels.end() ? mesh->get_default_boundary_id(primitive) : it->second;
			});
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
			const auto connectivity = unpack_connectivity(
				resources_.read_int_vector(higher_connectivity_path), resources_.read_long_vector(higher_offsets_path),
				resources_, higher_offsets_path);
			require_size(connectivity, mesh->n_elements(), resources_, higher_connectivity_path);
			mesh->attach_higher_order_nodes(resources_.read_matrix(higher_nodes_path), connectivity);
		}

		const std::string weights_path = child_path(path, "higher_order_weights");
		if (resources_.exists(weights_path))
		{
			const std::string offsets_path = child_path(path, "higher_order_weight_offsets");
			if (!resources_.exists(offsets_path))
				log_and_throw_error("{} requires higher_order_weight_offsets.", resources_.describe(weights_path));
			auto values = unpack_weights(
				resources_.read_double_vector(weights_path), resources_.read_long_vector(offsets_path),
				resources_, offsets_path);
			require_size(values, mesh->n_elements(), resources_, weights_path);
			mesh->set_cell_weights(values);
			mesh->set_is_rational(std::any_of(values.begin(), values.end(), [](const auto &entry) { return !entry.empty(); }));
		}
		return mesh;
	}

	SurfaceMesh MeshLoader::load_surface(const std::string &path) const
	{
		if (!resources_.exists(path))
			log_and_throw_error("Surface resource {} does not exist.", resources_.describe(path));
		SurfaceMesh result;
		if (!resources_.is_group(path))
		{
			if (!read_surface_mesh(resources_.materialize(path).string(), result.vertices, result.points, result.edges, result.faces))
				log_and_throw_error("Unable to decode surface mesh {}.", resources_.describe(path));
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
