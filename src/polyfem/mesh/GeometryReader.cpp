////////////////////////////////////////////////////////////////////////////////
#include "GeometryReader.hpp"

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MshReader.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Selection.hpp>

#include <Eigen/Core>

#include <igl/edges.h>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem
{
	using namespace utils;

	void log_and_throw_error(const std::string &msg)
	{
		logger().error(msg);
		throw std::runtime_error(msg);
	}

	void mesh::load_geometry(
		const json &geometry,
		const std::string &root_path,
		std::unique_ptr<Mesh> &mesh,
		Obstacle &obstacle,
		const std::vector<std::string> &_names,
		const std::vector<Eigen::MatrixXd> &_vertices,
		const std::vector<Eigen::MatrixXi> &_cells,
		const bool non_conforming)
	{
		// TODO: use these values
		assert(_names.empty());
		assert(_vertices.empty());
		assert(_cells.empty());

		if (geometry.empty())
			log_and_throw_error("Provided geometry is empty!");

		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells;
		std::vector<std::vector<int>> elements;
		std::vector<std::vector<double>> weights;
		size_t num_faces = 0;
		std::vector<std::shared_ptr<Selection>> surface_selections;
		std::vector<std::shared_ptr<Selection>> volume_selections;

		std::vector<json> geometries;
		// Note you can add more types here, just add them to geometries
		if (geometry.is_object())
		{
			geometries.push_back(geometry);
		}
		else if (geometry.is_array())
		{
			geometries = geometry.get<std::vector<json>>();
		}

		for (int i = 0; i < geometries.size(); i++)
		{
			json complete_geometry;
			apply_default_geometry_parameters(
				geometries[i], complete_geometry, fmt::format("/geometry[{}]", i));

			if (!complete_geometry["enabled"].get<bool>())
				continue;

			// TODO: handle loading obstacles
			if (complete_geometry["is_obstacle"].get<bool>())
				log_and_throw_error("Collision obstacles not implemented!");

			load_mesh(
				complete_geometry, root_path, vertices, cells, elements,
				weights, num_faces, surface_selections, volume_selections);
		}

		if (vertices.size() == 0)
			log_and_throw_error("No valid FEM meshes provided!");

		mesh = Mesh::create(vertices, cells, non_conforming);

		////////////////////////////////////////////////////////////////////////////

		// Only tris and tets
		if ((vertices.cols() == 2 && cells.cols() == 3) || (vertices.cols() == 3 && cells.cols() == 4))
		{
			mesh->attach_higher_order_nodes(vertices, elements);
			mesh->set_cell_weights(weights);
		}

		for (const auto &w : weights)
		{
			if (!w.empty())
			{
				mesh->set_is_rational(true);
				break;
			}
		}

		///////////////////////////////////////////////////////////////////////

		mesh->compute_boundary_ids([&](const size_t face_id, const RowVectorNd &p, bool is_boundary) {
			if (!is_boundary)
				return -1;

			for (const auto &selection : surface_selections)
				if (selection->inside(face_id, p))
					return selection->id(face_id);
			return -1;
		});

		// TODO: default boundary ids are all -1

		///////////////////////////////////////////////////////////////////////

		mesh->compute_body_ids([&](const size_t cell_id, const RowVectorNd &p) -> int {
			for (const auto &selection : volume_selections)
				if (selection->inside(cell_id, p))
					return selection->id(cell_id);
			return 0;
		});
	}

	///////////////////////////////////////////////////////////////////////////

	void mesh::load_mesh(
		const json &jmesh,
		const std::string &root_path,
		Eigen::MatrixXd &in_vertices,
		Eigen::MatrixXi &in_cells,
		std::vector<std::vector<int>> &in_elements,
		std::vector<std::vector<double>> &in_weights,
		size_t &num_faces,
		std::vector<std::shared_ptr<Selection>> &surface_selections,
		std::vector<std::shared_ptr<Selection>> &volume_selections)
	{
		if (!is_param_valid(jmesh, "mesh"))
			log_and_throw_error(fmt::format("Mesh {} is mising a \"mesh\" field!", jmesh));

		const std::string mesh_path = resolve_path(jmesh["mesh"], root_path);

		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells;
		std::vector<std::vector<int>> elements;
		std::vector<std::vector<double>> weights;
		std::vector<int> volume_ids;
		bool read_success = read_fem_mesh(
			mesh_path, vertices, cells, elements, weights, volume_ids);

		if (!read_success)
			// error already logged in read_fem_mesh()
			throw std::runtime_error(fmt::format("Unable to load mesh: {}", mesh_path));
		else if (vertices.size() == 0)
			log_and_throw_error(fmt::format("Mesh {} has zero vertices!", mesh_path));
		else if (cells.size() == 0)
			log_and_throw_error(fmt::format("Mesh {} has zero cells!", mesh_path));
		else if (in_vertices.cols() != 0 && in_vertices.cols() != vertices.cols())
			log_and_throw_error("Mixed dimension meshes is not implemented!");
		else if (in_cells.size() != 0 && in_cells.cols() != cells.cols())
			log_and_throw_error("Mixed tet and hex (tri and quad) meshes is not implemented!");

		////////////////////////////////////////////////////////////////////////////

		transform_mesh_from_json(jmesh["transformation"], vertices);

		////////////////////////////////////////////////////////////////////////////

		const size_t num_in_vertices = in_vertices.rows();
		const int dim = num_in_vertices == 0 ? (vertices.cols()) : (in_vertices.cols());
		assert(dim == 2 || dim == 3);
		in_vertices.conservativeResize(in_vertices.rows() + vertices.rows(), dim);
		in_vertices.bottomRows(vertices.rows()) = vertices;

		////////////////////////////////////////////////////////////////////////////

		const size_t num_in_cells = in_cells.rows();
		const int cell_cols = in_cells.cols() == 0 ? (cells.cols()) : (in_cells.cols());
		in_cells.conservativeResize(in_cells.rows() + cells.rows(), cell_cols);
		in_cells.bottomRows(cells.rows()) = cells.array() + num_in_vertices; // Shift vertex ids in cells

		////////////////////////////////////////////////////////////////////////////

		// TODO: Use force_linear_geometry does.
		// if (!jmesh["advanced"]["force_linear_geometry"].get<bool>())
		// {
		// Shift vertex ids in elements
		for (auto &element : elements)
			for (auto &id : element)
				id += num_in_vertices;
		in_elements.insert(in_elements.end(), elements.begin(), elements.end());
		in_weights.insert(in_weights.end(), weights.begin(), weights.end());
		// }

		///////////////////////////////////////////////////////////////////////

		const Selection::BBox bbox = {{vertices.colwise().minCoeff(), vertices.colwise().maxCoeff()}};

		if (!jmesh["point_selection"].is_null())
			logger().warn("Geometry point seleections are not implemented nor used!");

		if (!jmesh["curve_selection"].is_null())
			logger().warn("Geometry point seleections are not implemented nor used!");

		///////////////////////////////////////////////////////////////////////////

		const size_t num_local_faces = count_faces(dim, cells);
		append_selections(
			jmesh["surface_selection"], bbox, num_faces,
			num_faces + num_local_faces, surface_selections);
		num_faces += num_local_faces;

		////////////////////////////////////////////////////////////////////////////

		// Specified volume selection has priority over mesh's stored ids
		append_selections(
			jmesh["volume_selection"], bbox, num_in_cells,
			num_in_cells + cells.rows(), volume_selections);
		volume_selections.push_back(std::make_shared<SpecifiedSelection>(
			std::vector<int>(volume_ids.data(), volume_ids.data() + volume_ids.size()),
			num_in_cells, num_in_cells + cells.rows()));

		////////////////////////////////////////////////////////////////////////////

		if (jmesh["extract"].get<std::string>() != "volume")
			log_and_throw_error("Only volumetric elements are implemented for FEM meshes!");

		if (jmesh["n_refs"].get<int>() != 0)
		{
			log_and_throw_error("Option \"n_refs\" in geometry not implement yet!");
			if (jmesh["advanced"]["refinement_location"].get<double>() != 0.5)
				log_and_throw_error("Option \"refinement_location\" in geometry not implement yet!");
		}

		if (jmesh["advanced"]["min_component"].get<int>() != -1)
			log_and_throw_error("Option \"min_component\" in geometry not implement yet!");
	}

	///////////////////////////////////////////////////////////////////////////

	void mesh::apply_default_geometry_parameters(
		const json &geometry_in,
		json &geometry_out,
		const std::string &path_prefix)
	{
		geometry_out = R"({
			"type": "mesh",
			"mesh": null,
			"is_obstacle": false,
			"enabled": true,

			"transformation": {
				"translation": [0.0, 0.0, 0.0],
				"rotation": null,
				"rotation_mode": "xyz",
				"scale": [1.0, 1.0, 1.0],
				"dimensions": null
			},

			"extract": "volume",

			"point_selection": null,
			"curve_selection": null,
			"surface_selection": null,
			"volume_selection": null,

			"n_refs": 0,

			"advanced": {
				"force_linear_geometry": false,
				"refinement_location": 0.5,
				"normalize_mesh": false,
				"min_component": -1
			}
		})"_json;
		check_for_unknown_args(geometry_out, geometry_in, path_prefix);
		geometry_out.merge_patch(geometry_in);
	}

	///////////////////////////////////////////////////////////////////////////

	void mesh::transform_mesh_from_json(const json &transform, Eigen::MatrixXd &vertices)
	{
		const int dim = vertices.cols();
		assert(dim == 2 || dim == 3);

		// -----
		// Scale
		// -----

		RowVectorNd scale;
		if (transform["dimensions"].is_array()) // default is nullptr
		{
			VectorNd initial_dimensions =
				(vertices.colwise().maxCoeff() - vertices.colwise().minCoeff()).cwiseAbs();
			initial_dimensions =
				(initial_dimensions.array() == 0).select(1, initial_dimensions);

			scale = transform["dimensions"];
			const int scale_size = scale.size();
			scale.conservativeResize(dim);
			if (scale_size < dim)
				scale.tail(dim - scale_size).setZero();

			scale.array() /= initial_dimensions.array();
		}
		else if (transform["scale"].is_number())
		{
			scale.setConstant(dim, transform["scale"].get<double>());
		}
		else
		{
			assert(transform["scale"].is_array());
			scale = transform["scale"];
			const int scale_size = scale.size();
			scale.conservativeResize(dim);
			if (scale_size < dim)
				scale.tail(dim - scale_size).setZero();
		}

		vertices *= scale.asDiagonal();

		// ------
		// Rotate
		// ------

		// Rotate around the models origin NOT the bodies center of mass.
		// We could expose this choice as a "rotate_around" field.
		MatrixNd R = MatrixNd::Identity(dim, dim);
		if (!transform["rotation"].is_null())
		{
			if (dim == 2)
			{
				if (transform["rotation"].is_number())
					R = Eigen::Rotation2Dd(deg2rad(transform["rotation"].get<double>()))
							.toRotationMatrix();
				else
					log_and_throw_error("Invalid 2D rotation; 2D rotations can only be a angle in degrees.");
			}
			else if (dim == 3)
			{
				R = to_rotation_matrix(transform["rotation"], transform["rotation_mode"]);
			}
		}

		vertices *= R.transpose(); // (R*Vᵀ)ᵀ = V*Rᵀ

		// ---------
		// Translate
		// ---------

		RowVectorNd translation = transform["translation"];
		const int translation_size = translation.size();
		translation.conservativeResize(dim);
		if (translation_size < dim)
			translation.tail(dim - translation_size).setZero();

		vertices.rowwise() += translation;
	}

	////////////////////////////////////////////////////////////////////////////////

	void mesh::append_selections(
		const json &new_selections,
		const Selection::BBox &bbox,
		const size_t &start_element_id,
		const size_t &end_element_id,
		std::vector<std::shared_ptr<Selection>> &selections)
	{
		if (new_selections.is_number_integer())
		{
			selections.push_back(std::make_shared<UniformSelection>(
				new_selections.get<int>(), start_element_id, end_element_id));
		}
		else if (new_selections.is_string())
		{
			selections.push_back(std::make_shared<FileSelection>(
				new_selections.get<std::string>(), start_element_id, end_element_id));
		}
		else if (new_selections.is_object())
		{
			selections.push_back(Selection::build(
				new_selections, bbox, start_element_id, end_element_id));
		}
		else if (new_selections.is_array())
		{
			std::vector<json> new_selections_array = new_selections;
			for (const json &s : new_selections_array)
			{
				selections.push_back(Selection::build(
					s, bbox, start_element_id, end_element_id));
			}
		}
		else if (!new_selections.is_null())
		{
			log_and_throw_error(fmt::format("Invalid selections: {}", new_selections));
		}
	}

} // namespace polyfem