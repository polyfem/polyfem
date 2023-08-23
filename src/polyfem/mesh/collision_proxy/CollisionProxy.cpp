#include "CollisionProxy.hpp"

#include <polyfem/mesh/collision_proxy/UpsampleMesh.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::mesh
{
	void build_collision_proxy(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<LocalBoundary> &total_local_boundary,
		const double max_edge_length,
		Eigen::MatrixXd &proxy_vertices,
		Eigen::MatrixXi &proxy_faces,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries)
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
		// • the tessilations of all faces need to be stitched together
		//   - this means duplicate weights should be removed

		if (!mesh.is_conforming())
			log_and_throw_error("build_collision_proxy() is only implemented for conforming meshes!");

		std::vector<Eigen::Triplet<double>> displacement_map_entries_tmp;
		Eigen::MatrixXi proxy_faces_tmp;
		Eigen::MatrixXd proxy_vertices_tmp;

		Eigen::MatrixXd UV;
		Eigen::MatrixXi F_local;
		// TODO: use max_edge_length to determine the tessilation
		regular_grid_triangle_barycentric_coordinates(/*n=*/10, UV, F_local);

		for (const LocalBoundary &local_boundary : total_local_boundary)
		{
			if (local_boundary.type() != BoundaryType::TRI)
				log_and_throw_error("build_collision_proxy() is only implemented for triangles!");

			const basis::ElementBases elm = bases[local_boundary.element_id()];
			const basis::ElementBases g = geom_bases[local_boundary.element_id()];
			for (int fi = 0; fi < local_boundary.size(); fi++)
			{
				const int local_fid = local_boundary.local_primitive_id(fi);
				// const int global_fid = local_boundary.global_primitive_id(fi);
				// const Eigen::VectorXi nodes = elm.local_nodes_for_primitive(global_fid, mesh);

				// TODO: use the shape of f to determine the tessilation

				// Convert UV to appropirate UVW based on the local face id
				Eigen::MatrixXd UVW = Eigen::MatrixXd::Zero(UV.rows(), 3);
				switch (local_fid)
				{
				case 0:
					UVW.leftCols(2) = UV;
					break;
				case 1:
					UVW.col(0) = UV.col(0);
					UVW.col(2) = UV.col(1);
					break;
				case 2:
					UVW.leftCols(2) = UV;
					UVW.col(2) = 1 - UV.col(0).array() - UV.col(1).array();
					break;
				case 3:
					UVW.col(1) = UV.col(1);
					UVW.col(2) = UV.col(0);
					break;
				default:
					continue;
					log_and_throw_error("build_collision_proxy(): unknown local_fid={}", local_fid);
				}

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