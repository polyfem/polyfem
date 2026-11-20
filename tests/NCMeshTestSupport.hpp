#pragma once

#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
#include <polyfem/io/Checkpoint.hpp>
#include <polyfem/io/InputLoader.hpp>
#include <polyfem/utils/Logger.hpp>
#include <functional>

namespace polyfem::tests
{
	inline std::unique_ptr<mesh::Mesh> nc_base_mesh(int dimension)
	{
		Eigen::MatrixXd vertices;
		Eigen::MatrixXi cells;
		if (dimension == 2)
		{
			vertices.resize(4, 2);
			vertices << 0, 0, 1, 0, 1, 1, 0, 1;
			cells.resize(2, 3);
			cells << 0, 1, 2, 0, 2, 3;
		}
		else
		{
			vertices.resize(5, 3);
			vertices << .5, 0, 0, .5, 1, 0, .5, 0, 1, 0, 0, 0, 1, 0, 0;
			cells.resize(2, 4);
			cells << 0, 2, 1, 3, 0, 1, 2, 4;
		}
		mesh::MeshData data(vertices, cells);
		data.body_ids = {7, 9};
		data.geometry_ids = {4, 5};
		for (int i = 0; i < vertices.rows(); ++i)
			data.node_ids.push_back(100 + i);
		auto result = mesh::Mesh::create(data)->to_nonconforming();
		result->compute_boundary_ids([](size_t, const std::vector<int> &, const RowVectorNd &, bool boundary) { return boundary ? 11 : -1; });
		return result;
	}

	inline bool nc_barycenter_selection(const mesh::Mesh &m, int element) { return m.face_barycenter(element)(0) < .5; }
	inline bool nc_plane_selection(const mesh::Mesh &m, int element)
	{
		const Eigen::RowVector3d origin(.25, 0, 0), normal(1, 0, 0);
		for (int j = 0; j < m.n_cell_vertices(element); ++j)
			if ((m.point(m.cell_vertex(element, j)) - origin).dot(normal) < 0)
				return true;
		return false;
	}

	inline void nc_refine_elements(mesh::Mesh &m, const std::vector<int> &ids)
	{
		if (auto *nc = dynamic_cast<mesh::NCMesh2D *>(&m))
			nc->refine_elements(ids);
		else if (auto *nc = dynamic_cast<mesh::NCMesh3D *>(&m))
			nc->refine_elements(ids);
		else
			log_and_throw_error("Expected an NC test mesh.");
		m.prepare_mesh();
	}

	inline std::unique_ptr<mesh::Mesh> generate_nc_mesh(int dimension, int base_refs = 0, int local_refs = 1, int final_refs = 0,
														std::function<bool(const mesh::Mesh &, int)> selector = {})
	{
		auto result = nc_base_mesh(dimension);
		result->refine(base_refs, .5);
		result->prepare_mesh();
		if (!selector)
			selector = dimension == 2 ? nc_barycenter_selection : nc_plane_selection;
		for (int level = 0; level < local_refs; ++level)
		{
			std::vector<int> selected;
			for (int i = 0; i < result->n_elements(); ++i)
				if (selector(*result, i))
					selected.push_back(i);
			nc_refine_elements(*result, selected);
		}
		result->refine(final_refs, .5);
		result->prepare_mesh();
		return result;
	}

	/// Uses the production encoder for both ordinary input groups and checkpoint groups.
	inline void write_nc_bundle(const std::filesystem::path &path, const mesh::Mesh &m, const json &config)
	{
		io::CheckpointMetadata metadata;
		metadata.formulation = "Laplacian";
		metadata.dt = 1;
		io::CheckpointWriter writer(path, config, metadata);
		writer.write_mesh("/meshes/nc", m);
		writer.write_mesh("/checkpoint/meshes/active", m);
		writer.write_matrix("/checkpoint/state/solution", Eigen::MatrixXd::Zero(m.n_vertices(), 1));
		writer.finalize();
	}
} // namespace polyfem::tests
