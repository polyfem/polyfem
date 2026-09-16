#include "MeshData.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::mesh
{
	void MeshData::validate() const
	{
		if (vertices.rows() == 0 || (vertices.cols() != 2 && vertices.cols() != 3))
			log_and_throw_error("MeshData vertices must be a nonempty n x 2 or n x 3 matrix.");
		if (elements.rows() == 0 || elements.cols() < dimension() + 1)
			log_and_throw_error("MeshData elements have invalid dimensions.");

		for (int i = 0; i < elements.rows(); ++i)
		{
			int count = 0;
			bool found_padding = false;
			for (int j = 0; j < elements.cols(); ++j)
			{
				const int vertex = elements(i, j);
				if (vertex == -1)
				{
					found_padding = true;
					continue;
				}
				if (found_padding || vertex < 0 || vertex >= vertices.rows())
					log_and_throw_error("MeshData element {} contains invalid connectivity.", i);
				++count;
			}
			if (count < dimension() + 1)
				log_and_throw_error("MeshData element {} has too few vertices.", i);
		}

		const auto require_elements = [&](const size_t size, const std::string &name) {
			if (size != 0 && size != size_t(elements.rows()))
				log_and_throw_error("MeshData {} has {} entries; expected {}.", name, size, elements.rows());
		};
		require_elements(body_ids.size(), "body_ids");
		require_elements(geometry_ids.size(), "geometry_ids");
		require_elements(higher_order_connectivity.size(), "higher_order_connectivity");
		require_elements(higher_order_weights.size(), "higher_order_weights");
		if (!node_ids.empty() && node_ids.size() != size_t(vertices.rows()))
			log_and_throw_error("MeshData node_ids has {} entries; expected {}.", node_ids.size(), vertices.rows());

		if (boundary_ids.empty() != boundary_elements.empty())
			log_and_throw_error("MeshData boundary_elements and boundary_ids must be provided together.");
		if (!boundary_ids.empty() && boundary_ids.size() != boundary_elements.size())
			log_and_throw_error("MeshData boundary_elements and boundary_ids have different sizes.");
		for (const auto &element : boundary_elements)
		{
			if (element.size() < size_t(dimension()))
				log_and_throw_error("MeshData contains a boundary element with too few vertices.");
			for (const int vertex : element)
				if (vertex < 0 || vertex >= vertices.rows())
					log_and_throw_error("MeshData boundary connectivity references an invalid vertex.");
		}

		if (higher_order_connectivity.empty() != (higher_order_nodes.rows() == 0))
			log_and_throw_error("MeshData higher-order nodes and connectivity must be provided together.");
		if (!higher_order_connectivity.empty()
			&& (higher_order_nodes.cols() != dimension() || higher_order_nodes.rows() < vertices.rows()))
			log_and_throw_error("MeshData higher-order nodes have invalid dimensions.");
		for (const auto &connectivity : higher_order_connectivity)
			for (const int node : connectivity)
				if (node < 0 || node >= higher_order_nodes.rows())
					log_and_throw_error("MeshData higher-order connectivity references an invalid node.");
		if (!higher_order_weights.empty())
		{
			if (higher_order_connectivity.empty())
				log_and_throw_error("MeshData higher-order weights require higher-order connectivity.");
			for (int i = 0; i < higher_order_weights.size(); ++i)
				if (!higher_order_weights[i].empty()
					&& higher_order_weights[i].size() != higher_order_connectivity[i].size())
					log_and_throw_error("MeshData element {} has incompatible higher-order weights.", i);
		}

		if (has_polyhedral_topology())
		{
			if (dimension() != 3)
				log_and_throw_error("MeshData polyhedral topology requires three-dimensional vertices.");
			require_elements(cell_faces.size(), "cell_faces");
			require_elements(cell_face_orientations.size(), "cell_face_orientations");
			require_elements(cell_is_hex.size(), "cell_is_hex");
			if (cell_kernel_points.rows() != elements.rows() || cell_kernel_points.cols() != 3)
				log_and_throw_error("MeshData polyhedral cells require one 3D kernel point per element.");
			for (const auto &face : faces)
			{
				if (face.size() < 3)
					log_and_throw_error("MeshData contains a polyhedral face with fewer than three vertices.");
				for (const int vertex : face)
					if (vertex < 0 || vertex >= vertices.rows())
						log_and_throw_error("MeshData polyhedral face references an invalid vertex.");
			}
			for (int i = 0; i < elements.rows(); ++i)
			{
				if (cell_faces[i].size() != cell_face_orientations[i].size())
					log_and_throw_error("MeshData cell {} has inconsistent face orientations.", i);
				for (const int orientation : cell_face_orientations[i])
					if (orientation != 0 && orientation != 1)
						log_and_throw_error("MeshData cell {} has an invalid face orientation.", i);
				for (const int face : cell_faces[i])
					if (face < 0 || face >= faces.size())
						log_and_throw_error("MeshData cell {} references an invalid face.", i);
			}
		}
		else if (!cell_faces.empty() || !cell_face_orientations.empty() || !cell_is_hex.empty() || cell_kernel_points.size())
			log_and_throw_error("MeshData has incomplete polyhedral topology.");
	}
} // namespace polyfem::mesh
