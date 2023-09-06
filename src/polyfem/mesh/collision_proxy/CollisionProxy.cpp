#include "CollisionProxy.hpp"

#include <polyfem/mesh/collision_proxy/UpsampleMesh.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/edges.h>
#include <h5pp/h5pp.h>

namespace polyfem::mesh
{
	namespace
	{
		template <typename T>
		using RowMajorMatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<LocalBoundary> &total_local_boundary,
		const int n_bases,
		const int dim,
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

		std::vector<double> proxy_vertices_list;
		std::vector<int> proxy_faces_list;
		std::vector<Eigen::Triplet<double>> displacement_map_entries_tmp;

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

				const int offset = proxy_vertices_list.size() / dim;
				for (const double x : V_local.reshaped<Eigen::RowMajor>())
					proxy_vertices_list.push_back(x);
				for (const int i : F_local.reshaped<Eigen::RowMajor>())
					proxy_faces_list.push_back(i + offset);

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
			Eigen::Map<RowMajorMatrixX<double>>(proxy_vertices_list.data(), proxy_vertices_list.size() / dim, dim),
			Eigen::Map<RowMajorMatrixX<int>>(proxy_faces_list.data(), proxy_faces_list.size() / dim, dim),
			displacement_map_entries_tmp,
			proxy_vertices, proxy_faces, displacement_map_entries);
	}

	void load_collision_proxy(
		const std::string &mesh_filename,
		const std::string &weights_filename,
		const Eigen::VectorXi &in_node_to_node,
		const json &transformation,
		Eigen::MatrixXd &vertices,
		Eigen::VectorXi &codim_vertices,
		Eigen::MatrixXi &edges,
		Eigen::MatrixXi &faces,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries)
	{
#ifndef NDEBUG
		const size_t num_fe_nodes = in_node_to_node.size();
#endif

		Eigen::MatrixXi codim_edges;
		read_surface_mesh(mesh_filename, vertices, codim_vertices, codim_edges, faces);

		if (faces.size())
			igl::edges(faces, edges);

		utils::append_rows(edges, codim_edges);

		// TODO: transform the collision mesh in the same way the rest of the mesh is transformed
		// V = V * T.transpose();

		h5pp::File file(weights_filename, h5pp::FileAccess::READONLY);
		const std::array<long, 2> shape = file.readAttribute<std::array<long, 2>>("weight_triplets", "shape");
		assert(shape[0] == vertices.rows() && shape[1] == num_fe_nodes);
		Eigen::VectorXd values = file.readDataset<Eigen::VectorXd>("weight_triplets/values");
		Eigen::VectorXi rows = file.readDataset<Eigen::VectorXi>("weight_triplets/rows");
		Eigen::VectorXi cols = file.readDataset<Eigen::VectorXi>("weight_triplets/cols");
		assert(rows.maxCoeff() < vertices.rows());
		assert(cols.maxCoeff() < num_fe_nodes);

		// TODO: use these to build the in_node_to_node map
		// const Eigen::VectorXi in_ordered_vertices = file.exist("ordered_vertices") ? H5Easy::load<Eigen::VectorXi>(file, "ordered_vertices") : mesh->in_ordered_vertices();
		// const Eigen::MatrixXi in_ordered_edges = file.exist("ordered_edges") ? H5Easy::load<Eigen::MatrixXi>(file, "ordered_edges") : mesh->in_ordered_edges();
		// const Eigen::MatrixXi in_ordered_faces = file.exist("ordered_faces") ? H5Easy::load<Eigen::MatrixXi>(file, "ordered_faces") : mesh->in_ordered_faces();

		displacement_map_entries.clear();
		displacement_map_entries.reserve(values.size());

		assert(in_node_to_node.size() == num_fe_nodes);
		for (int i = 0; i < values.size(); i++)
		{
			// Rearrange the columns based on the FEM mesh node order
			displacement_map_entries.emplace_back(rows[i], in_node_to_node[cols[i]], values[i]);
		}

		// Transform the collision proxy
		std::array<RowVectorNd, 2> bbox;
		bbox[0] = vertices.colwise().minCoeff();
		bbox[1] = vertices.colwise().maxCoeff();

		MatrixNd A;
		VectorNd b;
		// TODO: pass correct unit scale
		construct_affine_transformation(
			/*unit_scale=*/1, transformation,
			(bbox[1] - bbox[0]).cwiseAbs().transpose(),
			A, b);
		vertices = (vertices * A.transpose()).rowwise() + b.transpose();
	}
} // namespace polyfem::mesh