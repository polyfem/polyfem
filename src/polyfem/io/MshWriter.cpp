#include "MshWriter.hpp"

#include <mshio/mshio.h>

namespace polyfem::io
{
	void MshWriter::write(
		const std::string &path,
		const mesh::Mesh &mesh,
		const bool binary)
	{
		mshio::MshSpec out;

		auto &format = out.mesh_format;
		format.version = "4.1";            // Only version "2.2" and "4.1" are supported.
		format.file_type = binary ? 1 : 0; // 0: ASCII, 1: binary.
		format.data_size = sizeof(size_t); // Size of data, defined as sizeof(size_t) = 8.

		auto &nodes = out.nodes;
		nodes.num_entity_blocks = 1;         // Number of node blocks.
		nodes.num_nodes = mesh.n_vertices(); // Total number of nodes.
		nodes.min_node_tag = 1;
		nodes.max_node_tag = mesh.n_vertices();
		nodes.entity_blocks.resize(1); // A std::vector of node blocks.
		{
			auto &block = nodes.entity_blocks[0];
			block.entity_dim = mesh.dimension();          // The dimension of the entity.
			block.entity_tag = 1;                         // The entity these nodes belongs to.
			block.parametric = 0;                         // 0: non-parametric, 1: parametric.
			block.num_nodes_in_block = mesh.n_vertices(); // The number of nodes in block.

			for (int i = 0; i < mesh.n_vertices(); ++i)
			{
				block.tags.push_back(i + 1); // A std::vector of unique, positive node tags.
				const auto p = mesh.point(i);
				block.data.push_back(p(0));
				block.data.push_back(p(1)); // A std::vector of coordinates (x,y,z,<u>,<v>,<w>,...)
				block.data.push_back(mesh.is_volume() ? p(2) : 0);
			}
		}

		auto &elements = out.elements;
		elements.num_entity_blocks = mesh.n_elements(); // Number of element blocks.
		elements.num_elements = mesh.n_elements();      // Total number of elmeents.
		elements.min_element_tag = 1;
		elements.max_element_tag = mesh.n_elements();
		elements.entity_blocks.resize(mesh.n_elements()); // A std::vector of element blocks.

		for (int i = 0; i < mesh.n_elements(); ++i)
		{
			auto &block = elements.entity_blocks[i];
			block.entity_dim = mesh.dimension(); // The dimension of the elements.
			block.entity_tag = 1;                // The entity these elements belongs to.
			const int n_local_v = mesh.n_cell_vertices(i);
			// only tet and tri for the moment
			assert(n_local_v == 3 || n_local_v == 4);

			block.element_type = n_local_v == 3 ? 2 : 4; // See element type table below.
			block.num_elements_in_block = 1;             // The number of elements in this block.
			block.data.push_back(i + 1);
			for (int j = 0; j < n_local_v; ++j)
				block.data.push_back(mesh.cell_vertex(i, j) + 1); // See more detail below.
		}

		mshio::save_msh(path, out);
	}
} // namespace polyfem::io
