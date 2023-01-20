#include "MshWriter.hpp"

#include <mshio/mshio.h>

namespace polyfem::io
{
	void MshWriter::write(
		const std::string &path,
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const std::vector<int> &body_ids,
		const bool is_volume,
		const bool binary)
	{
		assert(body_ids.size() == 0 || body_ids.size() == F.rows());

		mshio::MshSpec out;

		auto &format = out.mesh_format;
		format.version = "4.1";            // Only version "2.2" and "4.1" are supported.
		format.file_type = binary ? 1 : 0; // 0: ASCII, 1: binary.
		format.data_size = sizeof(size_t); // Size of data, defined as sizeof(size_t) = 8.

		auto &nodes = out.nodes;
		nodes.num_entity_blocks = 1; // Number of node blocks.
		nodes.num_nodes = V.rows();  // Total number of nodes.
		nodes.min_node_tag = 1;
		nodes.max_node_tag = V.rows();
		nodes.entity_blocks.resize(1); // A std::vector of node blocks.
		{
			auto &block = nodes.entity_blocks[0];
			block.entity_dim = V.cols();         // The dimension of the entity.
			block.entity_tag = 1;                // The entity these nodes belongs to.
			block.parametric = 0;                // 0: non-parametric, 1: parametric.
			block.num_nodes_in_block = V.rows(); // The number of nodes in block.

			for (int i = 0; i < V.rows(); ++i)
			{
				block.tags.push_back(i + 1); // A std::vector of unique, positive node tags.
				const auto p = V.row(i);
				block.data.push_back(p(0));
				block.data.push_back(p(1)); // A std::vector of coordinates (x,y,z,<u>,<v>,<w>,...)
				block.data.push_back(is_volume ? p(2) : 0);
			}
		}

		auto &elements = out.elements;
		elements.num_entity_blocks = F.rows(); // Number of element blocks.
		elements.num_elements = F.rows();      // Total number of elmeents.
		elements.min_element_tag = 1;
		elements.max_element_tag = F.rows();
		elements.entity_blocks.resize(F.rows()); // A std::vector of element blocks.

		for (int i = 0; i < F.rows(); ++i)
		{
			auto &block = elements.entity_blocks[i];
			block.entity_dim = V.cols(); // The dimension of the elements.
			block.entity_tag = 1;        // The entity these elements belongs to.
			const int n_local_v = F.cols();
			// only tet and tri for the moment
			assert(n_local_v == 3 || n_local_v == 4);

			block.element_type = n_local_v == 3 ? 2 : 4; // See element type table below.
			block.num_elements_in_block = 1;             // The number of elements in this block.
			block.data.push_back(i + 1);
			for (int j = 0; j < n_local_v; ++j)
				block.data.push_back(F(i, j) + 1); // See more detail below.

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
		// 	if (V.cols() == 2)
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
