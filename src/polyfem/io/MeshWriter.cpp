#include "MeshWriter.hpp"
#include <polyfem/mesh/MeshData.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <h5pp/h5pp.h>

namespace polyfem::io
{
	namespace
	{
		template <typename T>
		std::pair<std::vector<T>, std::vector<long>> pack_ragged(const std::vector<std::vector<T>> &rows)
		{
			std::pair<std::vector<T>, std::vector<long>> packed;
			packed.second.reserve(rows.size() + 1);
			packed.second.push_back(0);
			for (const auto &row : rows)
			{
				packed.first.insert(packed.first.end(), row.begin(), row.end());
				packed.second.push_back(packed.first.size());
			}
			return packed;
		}

		Eigen::MatrixXi pack_padded(const std::vector<std::vector<int>> &rows)
		{
			int width = 0;
			for (const auto &row : rows)
				width = std::max(width, int(row.size()));
			Eigen::MatrixXi packed = Eigen::MatrixXi::Constant(rows.size(), width, -1);
			for (int i = 0; i < rows.size(); ++i)
				for (int j = 0; j < rows[i].size(); ++j)
					packed(i, j) = rows[i][j];
			return packed;
		}
	} // namespace

	void write_mesh(h5pp::File &file, const std::string &group, const mesh::MeshData &data)
	{
		auto write_matrix = [&](const std::string &path, const Eigen::MatrixXd &value) { file.writeDataset(value, path); };
		auto write_int_matrix = [&](const std::string &path, const Eigen::MatrixXi &value) { file.writeDataset(value.cast<int64_t>(), path); };
		auto write_vector = [&](const std::string &path, const std::vector<double> &value) { file.writeDataset(value, path); };
		auto write_int_vector = [&](const std::string &path, const std::vector<int> &value) { file.writeDataset(value, path); };
		auto write_long_vector = [&](const std::string &path, const std::vector<long> &value) { file.writeDataset(value, path); };
		auto write_attribute = [&](const std::string &path, const std::string &name, const auto &value) { file.writeAttribute(value, path, name); };

		data.validate();
		write_matrix(group + "/vertices", data.vertices);
		write_int_matrix(group + "/cells", data.elements);
		write_attribute(group, "schema_version", mesh::MESH_SCHEMA_VERSION);
		write_attribute(group, "dimension", long(data.dimension()));
		write_attribute(group, "mesh_type", std::string("fem"));
		write_attribute(group, "elements_are_ordered", long(data.elements_are_ordered));
		if (data.nc)
		{
			const auto &nc = *data.nc;
			const std::string nc_group = group + "/nc";
			write_matrix(nc_group + "/vertices", nc.vertices);
			write_int_matrix(nc_group + "/cells", nc.cells);
			write_int_matrix(nc_group + "/ordered_cells", nc.ordered_cells);
			write_int_matrix(nc_group + "/cell_edges", nc.cell_edges);
			write_int_matrix(nc_group + "/children", nc.children);
			write_int_matrix(nc_group + "/element_state", nc.element_state);
			write_int_matrix(nc_group + "/edges", nc.edges);
			write_int_matrix(nc_group + "/midpoints", nc.midpoints);
			if (data.dimension() == 3)
			{
				write_int_matrix(nc_group + "/faces", nc.faces);
				write_int_matrix(nc_group + "/cell_faces", nc.cell_faces);
			}
			write_int_vector(nc_group + "/node_ids", nc.node_ids);
			write_int_vector(nc_group + "/refinement_history", nc.refinement_history);
			write_attribute(nc_group, "schema_version", long(1));
			write_attribute(nc_group, "label_flags", long(nc.label_flags));
		}

		if (!data.body_ids.empty())
			write_int_vector(group + "/body_ids", data.body_ids);
		if (!data.geometry_ids.empty())
			write_int_vector(group + "/geometry_ids", data.geometry_ids);
		if (!data.node_ids.empty())
			write_int_vector(group + "/node_ids", data.node_ids);
		if (!data.boundary_ids.empty())
		{
			write_int_matrix(group + "/boundary_elements", pack_padded(data.boundary_elements));
			write_int_vector(group + "/boundary_ids", data.boundary_ids);
		}
		if (!data.higher_order_connectivity.empty())
		{
			const auto packed = pack_ragged(data.higher_order_connectivity);
			write_matrix(group + "/higher_order_nodes", data.higher_order_nodes);
			write_int_vector(group + "/higher_order_connectivity", packed.first);
			write_long_vector(group + "/higher_order_offsets", packed.second);
		}
		if (!data.higher_order_weights.empty())
		{
			const auto packed = pack_ragged(data.higher_order_weights);
			write_vector(group + "/higher_order_weights", packed.first);
			write_long_vector(group + "/higher_order_weight_offsets", packed.second);
		}
		if (data.has_polyhedral_topology())
		{
			const auto faces = pack_ragged(data.faces);
			const auto cell_faces = pack_ragged(data.cell_faces);
			const auto orientations = pack_ragged(data.cell_face_orientations);
			write_int_vector(group + "/faces", faces.first);
			write_long_vector(group + "/face_offsets", faces.second);
			write_int_vector(group + "/cell_faces", cell_faces.first);
			write_long_vector(group + "/cell_face_offsets", cell_faces.second);
			write_int_vector(group + "/cell_face_orientations", orientations.first);
			std::vector<int> is_hex(data.cell_is_hex.size());
			std::transform(data.cell_is_hex.begin(), data.cell_is_hex.end(), is_hex.begin(), [](const bool value) { return int(value); });
			write_int_vector(group + "/cell_is_hex", is_hex);
			write_matrix(group + "/cell_kernel_points", data.cell_kernel_points);
		}
	}
} // namespace polyfem::io
