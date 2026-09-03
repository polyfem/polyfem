#include "MeshReader.hpp"

#include <polyfem/io/MshReader.hpp>
#include <polyfem/io/OBJReader.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>

#include <igl/read_triangle_mesh.h>

#include <algorithm>
#include <numeric>
#include <set>

namespace polyfem::mesh
{
	namespace
	{
		Eigen::MatrixXi padded_elements(const std::vector<std::vector<int>> &elements)
		{
			int width = 0;
			for (const auto &element : elements)
				width = std::max(width, int(element.size()));
			Eigen::MatrixXi result;
			result.setConstant(elements.size(), width, -1);
			for (int i = 0; i < elements.size(); ++i)
				for (int j = 0; j < elements[i].size(); ++j)
					result(i, j) = elements[i][j];
			return result;
		}

		Eigen::MatrixXd geogram_vertices(const GEO::Mesh &mesh, const int dimension)
		{
			Eigen::MatrixXd vertices(mesh.vertices.nb(), dimension);
			for (int i = 0; i < vertices.rows(); ++i)
				for (int d = 0; d < dimension; ++d)
					vertices(i, d) = mesh.vertices.point(i)[d];
			return vertices;
		}

		void finalize_surface(SurfaceMeshData &surface)
		{
			std::vector<bool> used(surface.vertices.rows(), false);
			for (int i = 0; i < surface.edges.rows(); ++i)
				for (int j = 0; j < surface.edges.cols(); ++j)
					used[surface.edges(i, j)] = true;
			for (int i = 0; i < surface.faces.rows(); ++i)
				for (int j = 0; j < surface.faces.cols(); ++j)
					used[surface.faces(i, j)] = true;
			const int count = std::count(used.begin(), used.end(), false);
			surface.points.resize(count);
			for (int i = 0, point = 0; i < used.size(); ++i)
				if (!used[i])
					surface.points[point++] = i;
		}
	} // namespace

	MeshData MeshReader::read_msh(const std::filesystem::path &path)
	{
		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells;
		std::vector<std::vector<int>> higher_order_connectivity;
		std::vector<std::vector<double>> weights;
		std::vector<int> body_ids;
		std::vector<std::vector<int>> boundary_elements;
		std::vector<int> boundary_ids;
		if (!io::MshReader::load(
				path.string(), vertices, cells, higher_order_connectivity, weights,
				body_ids, boundary_elements, boundary_ids))
			log_and_throw_error("Unable to decode MSH mesh {}.", path.string());

		MeshData data(std::move(vertices), std::move(cells));
		data.body_ids = std::move(body_ids);
		data.boundary_elements = std::move(boundary_elements);
		data.boundary_ids = std::move(boundary_ids);
		if ((data.dimension() == 2 && data.elements.cols() == 3)
			|| (data.dimension() == 3 && data.elements.cols() == 4))
		{
			data.higher_order_nodes = data.vertices;
			data.higher_order_connectivity = std::move(higher_order_connectivity);
			data.higher_order_weights = std::move(weights);
		}
		return data;
	}

	MeshData MeshReader::read_hybrid(std::istream &input, const std::string &description)
	{
		int vertex_count = 0;
		int face_count = 0;
		int encoded_cell_count = 0;
		if (!(input >> vertex_count >> face_count >> encoded_cell_count)
			|| vertex_count <= 0 || face_count <= 0 || encoded_cell_count <= 0 || encoded_cell_count % 3 != 0)
			log_and_throw_error("Invalid HYBRID header in {}.", description);
		const int cell_count = encoded_cell_count / 3;

		Eigen::MatrixXd vertices(vertex_count, 3);
		for (int i = 0; i < vertex_count; ++i)
			if (!(input >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2)))
				log_and_throw_error("HYBRID vertex data ended early in {}.", description);

		std::vector<std::vector<int>> faces(face_count);
		for (auto &face : faces)
		{
			int size = 0;
			if (!(input >> size) || size < 3)
				log_and_throw_error("Invalid HYBRID face in {}.", description);
			face.resize(size);
			for (int &vertex : face)
				if (!(input >> vertex))
					log_and_throw_error("HYBRID face data ended early in {}.", description);
		}

		std::vector<std::vector<int>> cell_faces(cell_count);
		std::vector<std::vector<int>> orientations(cell_count);
		std::vector<std::vector<int>> elements(cell_count);
		for (int i = 0; i < cell_count; ++i)
		{
			int face_size = 0;
			if (!(input >> face_size) || face_size <= 0)
				log_and_throw_error("Invalid HYBRID cell in {}.", description);
			cell_faces[i].resize(face_size);
			for (int &face : cell_faces[i])
				if (!(input >> face) || face < 0 || face >= face_count)
					log_and_throw_error("Invalid HYBRID cell face in {}.", description);

			int orientation_size = 0;
			if (!(input >> orientation_size) || orientation_size != face_size)
				log_and_throw_error("Invalid HYBRID face orientations in {}.", description);
			orientations[i].resize(orientation_size);
			for (int &orientation : orientations[i])
				if (!(input >> orientation))
					log_and_throw_error("HYBRID orientation data ended early in {}.", description);

			std::set<int> vertices_in_cell;
			for (const int face : cell_faces[i])
				vertices_in_cell.insert(faces[face].begin(), faces[face].end());
			elements[i].assign(vertices_in_cell.begin(), vertices_in_cell.end());
		}

		std::vector<bool> cell_is_hex(cell_count);
		for (int i = 0; i < cell_count; ++i)
		{
			int value = 0;
			if (!(input >> value))
				log_and_throw_error("HYBRID cell-type data ended early in {}.", description);
			cell_is_hex[i] = value != 0;
		}

		Eigen::MatrixXd kernels(cell_count, 3);
		kernels.setZero();
		std::string marker;
		if (input >> marker)
		{
			int count = 0;
			if (marker != "KERNEL" || !(input >> count) || count != cell_count)
				log_and_throw_error("Invalid HYBRID kernel block in {}.", description);
			for (int i = 0; i < cell_count; ++i)
				if (!(input >> kernels(i, 0) >> kernels(i, 1) >> kernels(i, 2)))
					log_and_throw_error("HYBRID kernel data ended early in {}.", description);
		}
		else
		{
			input.clear();
			for (int i = 0; i < cell_count; ++i)
			{
				for (const int vertex : elements[i])
					kernels.row(i) += vertices.row(vertex);
				kernels.row(i) /= elements[i].size();
			}
		}

		MeshData data(std::move(vertices), padded_elements(elements));
		data.faces = std::move(faces);
		data.cell_faces = std::move(cell_faces);
		data.cell_face_orientations = std::move(orientations);
		data.cell_is_hex = std::move(cell_is_hex);
		data.cell_kernel_points = std::move(kernels);
		return data;
	}

	MeshData MeshReader::read_geogram(const std::filesystem::path &path)
	{
		GEO::Mesh mesh;
		if (!GEO::mesh_load(path.string(), mesh))
			log_and_throw_error("Unable to decode mesh {} with Geogram.", path.string());
		return from_geogram(mesh);
	}

	MeshData MeshReader::from_geogram(GEO::Mesh &mesh)
	{
		if (is_planar(mesh))
		{
			std::vector<std::vector<int>> elements(mesh.facets.nb());
			for (int i = 0; i < elements.size(); ++i)
			{
				elements[i].resize(mesh.facets.nb_vertices(i));
				for (int j = 0; j < elements[i].size(); ++j)
					elements[i][j] = mesh.facets.vertex(i, j);
			}
			return MeshData(geogram_vertices(mesh, 2), padded_elements(elements));
		}

		if (mesh.cells.nb() != 0)
		{
			std::vector<std::vector<int>> elements(mesh.cells.nb());
			for (int i = 0; i < elements.size(); ++i)
			{
				elements[i].resize(mesh.cells.nb_vertices(i));
				for (int j = 0; j < elements[i].size(); ++j)
					elements[i][j] = mesh.cells.vertex(i, j);
			}
			return MeshData(geogram_vertices(mesh, 3), padded_elements(elements));
		}

		if (mesh.facets.nb() == 0)
			log_and_throw_error("Geogram mesh contains neither cells nor facets.");
		std::vector<std::vector<int>> faces(mesh.facets.nb());
		std::set<int> used_vertices;
		for (int i = 0; i < faces.size(); ++i)
		{
			faces[i].resize(mesh.facets.nb_vertices(i));
			for (int j = 0; j < faces[i].size(); ++j)
			{
				faces[i][j] = mesh.facets.vertex(i, j);
				used_vertices.insert(faces[i][j]);
			}
		}
		std::vector<std::vector<int>> elements(1);
		elements[0].assign(used_vertices.begin(), used_vertices.end());
		MeshData data(geogram_vertices(mesh, 3), padded_elements(elements));
		data.faces = std::move(faces);
		data.cell_faces.resize(1);
		data.cell_faces[0].resize(data.faces.size());
		std::iota(data.cell_faces[0].begin(), data.cell_faces[0].end(), 0);
		data.cell_face_orientations = {std::vector<int>(data.faces.size(), 1)};
		data.cell_is_hex = {false};
		data.cell_kernel_points.resize(1, 3);
		const int last = mesh.vertices.nb() - 1;
		if (used_vertices.find(last) == used_vertices.end())
			data.cell_kernel_points.row(0) = data.vertices.row(last);
		else
		{
			data.cell_kernel_points.setZero();
			for (const int vertex : elements[0])
				data.cell_kernel_points.row(0) += data.vertices.row(vertex);
			data.cell_kernel_points.row(0) /= elements[0].size();
		}
		return data;
	}

	SurfaceMeshData MeshReader::read_msh_surface(const std::filesystem::path &path)
	{
		SurfaceMeshData result;
		Eigen::MatrixXi cells;
		std::vector<std::vector<int>> elements;
		std::vector<std::vector<double>> weights;
		std::vector<int> body_ids;
		if (!io::MshReader::load(path.string(), result.vertices, cells, elements, weights, body_ids))
			log_and_throw_error("Unable to decode MSH surface {}.", path.string());

		if (cells.cols() == 1)
			result.points = cells;
		else if (cells.cols() == 2)
			result.edges = cells;
		else if (cells.cols() == 3)
			result.faces = cells;
		else if (cells.cols() == 4 && result.vertices.cols() == 3)
		{
			Eigen::MatrixXd surface_vertices;
			extract_triangle_surface_from_tets(
				result.vertices, cells, surface_vertices, result.faces);
			result.vertices = std::move(surface_vertices);
		}
		else
			log_and_throw_error("Unsupported MSH surface cell type in {}.", path.string());

		finalize_surface(result);
		return result;
	}

	SurfaceMeshData MeshReader::read_obj_surface(const std::filesystem::path &path)
	{
		SurfaceMeshData result;
		if (!io::OBJReader::read(path.string(), result.vertices, result.edges, result.faces))
			log_and_throw_error("Unable to decode OBJ surface {}.", path.string());
		finalize_surface(result);
		return result;
	}

	bool MeshReader::read_triangle_surface(
		const std::filesystem::path &path,
		SurfaceMeshData &result)
	{
		result = SurfaceMeshData();
		if (!igl::read_triangle_mesh(path.string(), result.vertices, result.faces))
			return false;
		finalize_surface(result);
		return true;
	}

	SurfaceMeshData MeshReader::read_geogram_surface(const std::filesystem::path &path)
	{
		GEO::Mesh mesh;
		if (!GEO::mesh_load(path.string(), mesh))
			log_and_throw_error("Unable to decode surface {} with Geogram.", path.string());
		if (mesh.facets.nb() == 0)
			log_and_throw_error("Surface mesh {} contains no facets.", path.string());

		SurfaceMeshData result;
		result.vertices = geogram_vertices(mesh, is_planar(mesh) ? 2 : 3);
		const int width = mesh.facets.nb_vertices(0);
		result.faces.resize(mesh.facets.nb(), width);
		for (int i = 0; i < result.faces.rows(); ++i)
		{
			if (mesh.facets.nb_vertices(i) != width)
				log_and_throw_error("Surface mesh {} has mixed facet sizes.", path.string());
			for (int j = 0; j < width; ++j)
				result.faces(i, j) = mesh.facets.vertex(i, j);
		}
		finalize_surface(result);
		return result;
	}
} // namespace polyfem::mesh
