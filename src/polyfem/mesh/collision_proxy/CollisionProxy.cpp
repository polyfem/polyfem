#include "CollisionProxy.hpp"

#include <polyfem/mesh/collision_proxy/UpsampleMesh.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::mesh
{
	namespace
	{
		/// @brief Convert from 2D barycentric coordinates (BCs) of a triangle to 3D BCs in a tet.
		/// @param uv 2D BCs of a triangular face
		/// @param local_fid which face do the 2D BCs coorespond to
		/// @return 3D BCs in the tet
		Eigen::MatrixXd uv_to_uvw(const Eigen::MatrixXd &uv, const int local_fid)
		{
			assert(uv.cols() == 2);

			Eigen::MatrixXd uvw = Eigen::MatrixXd::Zero(uv.rows(), 3);
			// u * A + v * B + w * C + (1 - u - v - w) * D
			switch (local_fid)
			{
			case 0:
				uvw.col(0) = uv.col(1);
				uvw.col(1) = uv.col(0);
				break;
			case 1:
				uvw.col(0) = uv.col(0);
				uvw.col(2) = uv.col(1);
				break;
			case 2:
				uvw.leftCols(2) = uv;
				uvw.col(2) = 1 - uv.col(0).array() - uv.col(1).array();
				break;
			case 3:
				uvw.col(1) = uv.col(1);
				uvw.col(2) = uv.col(0);
				break;
			default:
				log_and_throw_error("build_collision_proxy(): unknown local_fid={}", local_fid);
			}
			return uvw;
		}

		Eigen::MatrixXd extract_face_vertices(
			const basis::ElementBases &element, const int local_fid)
		{
			Eigen::MatrixXd UV(3, 2);
			// u * A + b * B + (1-u-v) * C
			UV.row(0) << 1, 0;
			UV.row(1) << 0, 1;
			UV.row(2) << 0, 0;
			const Eigen::MatrixXd UVW = uv_to_uvw(UV, local_fid);

			Eigen::MatrixXd V;
			element.eval_geom_mapping(UVW, V);

			return V;
		}
	} // namespace

	void build_collision_proxy(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<LocalBoundary> &total_local_boundary,
		const double max_edge_length,
		Eigen::MatrixXd &proxy_vertices,
		Eigen::MatrixXi &proxy_faces,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries,
		const CollisionProxyTessellation tessellation)
	{
		// for each boundary element (f):
		//     tessilate f with triangles of max edge length (fₜ)
		//     for each node (x) of fₜ with global index (i):
		//         for each basis (ϕⱼ) in f's parent element:
		//             set Vᵢ = x where Vᵢ is the i-th proxy vertex
		//             set W(i, j) = ϕⱼ(g⁻¹(x)) where g is the geometry mapping of f
		// caveats:
		// • if x is provided in parametric coordinates, we can skip evaluating g⁻¹
		//   - but Vᵢ = g(x) instead
		// • the tessellations of all faces need to be stitched together
		//   - this means duplicate weights should be removed

		if (!mesh.is_conforming())
			log_and_throw_error("build_collision_proxy() is only implemented for conforming meshes!");

		std::vector<Eigen::Triplet<double>> displacement_map_entries_tmp;
		Eigen::MatrixXi proxy_faces_tmp;
		Eigen::MatrixXd proxy_vertices_tmp;

		Eigen::MatrixXd UV;
		Eigen::MatrixXi F_local;
		if (tessellation == CollisionProxyTessellation::REGULAR)
		{
			// TODO: use max_edge_length to determine the tessellation
			regular_grid_triangle_barycentric_coordinates(/*n=*/10, UV, F_local);
		}

		for (const LocalBoundary &local_boundary : total_local_boundary)
		{
			if (local_boundary.type() != BoundaryType::TRI)
				log_and_throw_error("build_collision_proxy() is only implemented for triangles!");

			const basis::ElementBases elm = bases[local_boundary.element_id()];
			const basis::ElementBases g = geom_bases[local_boundary.element_id()];
			for (int fi = 0; fi < local_boundary.size(); fi++)
			{
				const int local_fid = local_boundary.local_primitive_id(fi);

				if (tessellation == CollisionProxyTessellation::IRREGULAR)
				{
					// Use the shape of f to determine the tessellation
					const Eigen::MatrixXd node_positions = extract_face_vertices(g, local_fid);
					irregular_triangle_barycentric_coordinates(
						node_positions.row(0), node_positions.row(1), node_positions.row(2),
						max_edge_length, UV, F_local);
				}

				// Convert UV to appropirate UVW based on the local face id
				Eigen::MatrixXd UVW = uv_to_uvw(UV, local_fid);

				Eigen::MatrixXd V_local;
				g.eval_geom_mapping(UVW, V_local);
				assert(V_local.rows() == UV.rows());

				const int offset = proxy_vertices_tmp.rows();
				utils::append_rows(proxy_vertices_tmp, V_local);
				utils::append_rows(proxy_faces_tmp, F_local.array() + offset);

				for (const basis::Basis &basis : elm.bases)
				{
					assert(basis.global().size() == 1);
					const int basis_id = basis.global()[0].index;

					const Eigen::MatrixXd basis_values = basis(UVW);

					for (int i = 0; i < basis_values.size(); i++)
					{
						displacement_map_entries_tmp.emplace_back(
							offset + i, basis_id, basis_values(i));
					}
				}
			}
		}

		// stitch collision proxy together
		stitch_mesh(
			proxy_vertices_tmp, proxy_faces_tmp, displacement_map_entries_tmp,
			proxy_vertices, proxy_faces, displacement_map_entries);
	}
} // namespace polyfem::mesh