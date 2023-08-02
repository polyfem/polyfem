#include "MshWriter.hpp"

#include <mshio/mshio.h>

namespace polyfem::io
{
	void MshWriter::write(
		const std::string &path,
		const mesh::Mesh &mesh,
		const bool binary)
	{
		Eigen::MatrixXd points(mesh.n_vertices(), mesh.dimension());
		for (int i = 0; i < mesh.n_vertices(); ++i)
			points.col(i) = mesh.point(i);

		std::vector<std::vector<int>> cells(mesh.n_elements());
		for (int i = 0; i < mesh.n_elements(); ++i)
		{
			cells[i].resize(mesh.n_cell_vertices(i));
			for (int j = 0; j < cells[i].size(); ++j)
				cells[i][j] = mesh.cell_vertex(i, j);
		}

		return write(path, points, cells, mesh.get_body_ids(), mesh.is_volume(), binary);
	}

	void MshWriter::write(
		const std::string &path,
		const Eigen::MatrixXd &points,
		const Eigen::MatrixXi &cells,
		const std::vector<int> &body_ids,
		const bool is_volume,
		const bool binary)
	{
		std::vector<std::vector<int>> cells_vector(cells.rows(), std::vector<int>(cells.cols()));
		for (int i = 0; i < cells.rows(); ++i)
			for (int j = 0; j < cells.cols(); ++j)
				cells_vector[i][j] = cells(i, j);

		return write(path, points, cells_vector, body_ids, is_volume, binary);
	}

	void MshWriter::write(
		const std::string &path,
		const Eigen::MatrixXd &points,
		const std::vector<std::vector<int>> &cells,
		const std::vector<int> &body_ids,
		const bool is_volume,
		const bool binary)
	{
		assert(body_ids.size() == 0 || body_ids.size() == cells.size());

		mshio::MshSpec out;

		auto &format = out.mesh_format;
		format.version = "4.1";            // Only version "2.2" and "4.1" are supported.
		format.file_type = binary ? 1 : 0; // 0: ASCII, 1: binary.
		format.data_size = sizeof(size_t); // Size of data, defined as sizeof(size_t) = 8.

		auto &nodes = out.nodes;
		nodes.num_entity_blocks = 1;     // Number of node blocks.
		nodes.num_nodes = points.rows(); // Total number of nodes.
		nodes.min_node_tag = 1;
		nodes.max_node_tag = points.rows();
		nodes.entity_blocks.resize(1); // A std::vector of node blocks.
		{
			auto &block = nodes.entity_blocks[0];
			block.entity_dim = points.cols();         // The dimension of the entity.
			block.entity_tag = 1;                     // The entity these nodes belongs to.
			block.parametric = 0;                     // 0: non-parametric, 1: parametric.
			block.num_nodes_in_block = points.rows(); // The number of nodes in block.

			for (int i = 0; i < points.rows(); ++i)
			{
				block.tags.push_back(i + 1); // A std::vector of unique, positive node tags.
				const auto p = points.row(i);
				block.data.push_back(p(0));
				block.data.push_back(p(1)); // A std::vector of coordinates (x,y,z,<u>,<points>,<w>,...)
				block.data.push_back(is_volume ? p(2) : 0);
			}
		}

		auto &elements = out.elements;
		elements.num_entity_blocks = cells.size(); // Number of element blocks.
		elements.num_elements = cells.size();      // Total number of elmeents.
		elements.min_element_tag = 1;
		elements.max_element_tag = cells.size();
		elements.entity_blocks.resize(cells.size()); // A std::vector of element blocks.

		for (int i = 0; i < cells.size(); ++i)
		{
			auto &block = elements.entity_blocks[i];
			block.entity_dim = points.cols(); // The dimension of the elements.
			block.entity_tag = 1;             // The entity these elements belongs to.
			const int n_local_v = cells[i].size();
			// only tet and tri for the moment
			assert(n_local_v == 3 || n_local_v == 4);

			block.element_type = n_local_v == 3 ? 2 : 4; // See element type table below.
			block.num_elements_in_block = 1;             // The number of elements in this block.
			block.data.push_back(i + 1);
			for (int j = 0; j < n_local_v; ++j)
				block.data.push_back(cells[i][j] + 1); // See more detail below.

			// block.entity_tag = body_ids.empty() ? 0 : body_ids[i];
		}

		// Add physical groups based on body ids
		// std::vector<int> unique_body_ids = body_ids;
		// if (unique_body_ids.size() == 0)
		// 	unique_body_ids.push_back(0); // default body id
		// std::sort(unique_body_ids.begin(), unique_body_ids.end());
		// unique_body_ids.erase(std::unique(unique_body_ids.begin(), unique_body_ids.end()), unique_body_ids.end());
		// for (const int body_id : body_ids)
		// {
		// 	if (points.cols() == 2)
		// 	{
		// 		out.entities.surfaces.emplace_back();
		// 		out.entities.surfaces.back().tag = body_id;
		// 		out.entities.surfaces.back().physical_group_tags.push_back(body_id);
		// 	}
		// 	else
		// 	{
		// 		out.entities.volumes.emplace_back();
		// 		out.entities.volumes.back().tag = body_id;
		// 		out.entities.volumes.back().physical_group_tags.push_back(body_id);
		// 	}
		// }

		mshio::save_msh(path, out);
	}
} // namespace polyfem::io
