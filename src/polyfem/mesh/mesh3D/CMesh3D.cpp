#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/CellConnectivity.hpp>
#include <polyfem/mesh/mesh3D/MeshProcessing3D.hpp>
#include <polyfem/mesh/MeshUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/barycentric_coordinates.h>

#include <unordered_map>

using namespace polyfem::utils;

namespace polyfem
{
	namespace mesh
	{
		void CMesh3D::remove_elements(const std::vector<bool> &keep)
		{
			assert(keep.size() == n_cells());

			auto face_key = [this](const int f) {
				std::vector<int> key(n_face_vertices(f));
				for (int lv = 0; lv < key.size(); ++lv)
					key[lv] = face_vertex(f, lv);
				std::sort(key.begin(), key.end());
				return key;
			};
			auto edge_key = [this](const int e) {
				return std::pair<int, int>(
					std::min(edge_vertex(e, 0), edge_vertex(e, 1)),
					std::max(edge_vertex(e, 0), edge_vertex(e, 1)));
			};

			std::unordered_map<std::vector<int>, int, utils::HashVector> old_boundary_ids;
			std::unordered_map<std::vector<int>, FaceNodes, utils::HashVector> old_face_nodes;
			for (int f = 0; f < n_faces(); ++f)
			{
				const auto key = face_key(f);
				if (has_boundary_ids())
					old_boundary_ids[key] = boundary_ids_[f];
				if (f < face_nodes_.size())
					old_face_nodes[key] = face_nodes_[f];
			}

			std::unordered_map<std::pair<int, int>, EdgeNodes, utils::HashPair> old_edge_nodes;
			for (int e = 0; e < n_edges() && e < edge_nodes_.size(); ++e)
				old_edge_nodes[edge_key(e)] = edge_nodes_[e];

			if (cell_nodes_.size() == keep.size())
			{
				std::vector<CellNodes> filtered;
				for (int c = 0; c < keep.size(); ++c)
					if (keep[c])
						filtered.push_back(cell_nodes_[c]);
				cell_nodes_ = std::move(filtered);
			}
			filter_element_data(keep);

			std::vector<bool> used_faces(mesh_.n_faces(), false);
			for (int c = 0; c < keep.size(); ++c)
				if (keep[c])
					for (const int f : mesh_.cell_fs[c])
						used_faces[f] = true;

			std::vector<int> old_to_new_face(mesh_.n_faces(), -1);
			JaggedArray<uint32_t> face_vs;
			for (int f = 0; f < used_faces.size(); ++f)
			{
				if (!used_faces[f])
					continue;
				old_to_new_face[f] = face_vs.size();
				face_vs.push_back(mesh_.face_vs[f]);
			}

			JaggedArray<uint32_t> cell_vs, cell_input_vs, cell_fs;
			JaggedArray<uint8_t> cell_fs_flag;
			std::vector<uint8_t> cell_hex;
			std::vector<std::array<double, 3>> cell_kernel;
			std::vector<uint32_t> fs;
			for (int c = 0; c < keep.size(); ++c)
			{
				if (!keep[c])
					continue;
				cell_vs.push_back(mesh_.cell_vs[c]);
				cell_input_vs.push_back(mesh_.cell_input_vs[c]);
				fs.clear();
				for (const auto f : mesh_.cell_fs[c])
					fs.push_back(old_to_new_face[f]);
				cell_fs.push_back(fs);
				cell_fs_flag.push_back(mesh_.cell_fs_flag[c]);
				cell_hex.push_back(mesh_.cell_hex[c]);
				cell_kernel.push_back(mesh_.cell_kernel[c]);
			}
			mesh_.face_vs = std::move(face_vs);
			mesh_.cell_vs = std::move(cell_vs);
			mesh_.cell_input_vs = std::move(cell_input_vs);
			mesh_.cell_fs = std::move(cell_fs);
			mesh_.cell_fs_flag = std::move(cell_fs_flag);
			mesh_.cell_hex = std::move(cell_hex);
			mesh_.cell_kernel = std::move(cell_kernel);

			Navigation3D::prepare_mesh(mesh_);

			if (!old_boundary_ids.empty())
			{
				boundary_ids_.resize(n_faces());
				for (int f = 0; f < n_faces(); ++f)
				{
					const auto it = old_boundary_ids.find(face_key(f));
					boundary_ids_[f] = it == old_boundary_ids.end() ? get_default_boundary_id(f) : it->second;
				}
			}
			else
				boundary_ids_.clear();

			face_nodes_.clear();
			if (!old_face_nodes.empty())
			{
				face_nodes_.resize(n_faces());
				for (int f = 0; f < n_faces(); ++f)
				{
					const auto it = old_face_nodes.find(face_key(f));
					if (it != old_face_nodes.end())
						face_nodes_[f] = it->second;
				}
			}

			edge_nodes_.clear();
			if (!old_edge_nodes.empty())
			{
				edge_nodes_.resize(n_edges());
				for (int e = 0; e < n_edges(); ++e)
				{
					const auto it = old_edge_nodes.find(edge_key(e));
					if (it != old_edge_nodes.end())
						edge_nodes_[e] = it->second;
				}
			}

			in_ordered_edges_.resize(n_edges(), 2);
			for (int e = 0; e < n_edges(); ++e)
				in_ordered_edges_.row(e) << edge_vertex(e, 0), edge_vertex(e, 1);
			int max_face_size = 0;
			for (int f = 0; f < n_faces(); ++f)
				max_face_size = std::max(max_face_size, n_face_vertices(f));
			in_ordered_faces_.setConstant(n_faces(), max_face_size, -1);
			for (int f = 0; f < n_faces(); ++f)
				for (int lv = 0; lv < n_face_vertices(f); ++lv)
					in_ordered_faces_(f, lv) = face_vertex(f, lv);
			compute_elements_tag();
		}

		void CMesh3D::refine(const int n_refinement, const double t)
		{
			if (n_refinement <= 0)
			{
				return;
			}

			// TODO refine high order mesh!
			clear_data_lost_by_refinement();
			if (mesh_.type == MeshType::TET)
			{
				MeshProcessing3D::refine_red_refinement_tet(mesh_, n_refinement);
			}
			else
			{
				for (size_t i = 0; i < elements_tag().size(); ++i)
				{
					if (elements_tag()[i] == ElementType::INTERIOR_POLYTOPE || elements_tag()[i] == ElementType::BOUNDARY_POLYTOPE)
						mesh_.cell_hex[i] = false;
				}

				bool reverse_grow = false;
				std::vector<int> parent_nodes;
				MeshProcessing3D::refine_catmul_clark_polar(mesh_, n_refinement, reverse_grow, parent_nodes);
			}

			Navigation3D::prepare_mesh(mesh_);
			compute_elements_tag();

			in_ordered_vertices_ = Eigen::VectorXi::LinSpaced(n_vertices(), 0, n_vertices() - 1);
			assert(in_ordered_vertices_[0] == 0);
			assert(in_ordered_vertices_[1] == 1);
			assert(in_ordered_vertices_[2] == 2);
			assert(in_ordered_vertices_[in_ordered_vertices_.size() - 1] == n_vertices() - 1);

			in_ordered_edges_.resize(mesh_.n_edges(), 2);

			for (int e = 0; e < (int)mesh_.n_edges(); ++e)
			{
				assert(mesh_.edge_vs[e].size() == 2);
				for (int lv = 0; lv < 2; ++lv)
				{
					in_ordered_edges_(e, lv) = mesh_.edge_vs[e][lv];
				}
			}
			assert(in_ordered_edges_.size() > 0);

			in_ordered_faces_.resize(mesh_.n_faces(), mesh_.face_vs[0].size());

			for (int f = 0; f < (int)mesh_.n_faces(); ++f)
			{
				assert(in_ordered_faces_.cols() == mesh_.face_vs[f].size());

				for (int lv = 0; lv < in_ordered_faces_.cols(); ++lv)
				{
					in_ordered_faces_(f, lv) = mesh_.face_vs[f][lv];
				}
			}
			assert(in_ordered_faces_.size() > 0);
		}

		// Build the internal topology from standard cells (tets, pyramids, prisms, hexes).
		bool CMesh3D::build_storage_from_cells(const Eigen::MatrixXd &V, const std::vector<StandardCell> &cells)
		{
			edge_nodes_.clear();
			face_nodes_.clear();
			cell_nodes_.clear();

			assert(V.cols() == 3);

			// Set vertices
			const int nv = V.rows();
			mesh_ = Mesh3DStorage();
			mesh_.points.resize(3, nv);
			for (int i = 0; i < nv; ++i)
			{
				mesh_.points(0, i) = V(i, 0);
				mesh_.points(1, i) = V(i, 1);
				mesh_.points(2, i) = V(i, 2);
			}

			const bool all_tets = std::all_of(cells.begin(), cells.end(), [](const StandardCell &c) {
				return c.type == StandardCellType::TET;
			});

			// Set cells
			// MeshData guarantees at least one cell.
			if (cells.empty())
				log_and_throw_error("A volume mesh needs at least one cell.");
			else
			{
				const std::vector<std::array<int, 6>> adjacent = connect_cells(cells, nv);

				// Index of the cell facets (c, lf).
				std::vector<int> first_cell_facet(cells.size() + 1, 0);
				for (int c = 0; c < cells.size(); ++c)
					first_cell_facet[c + 1] = first_cell_facet[c] + cells[c].n_facets();

				// Cell facet of the neighbor across (c, lf) that is glued to c.
				auto opposite_cell_facet = [&](int c, int lf) {
					const int c2 = adjacent[c][lf];
					if (c2 == -1)
					{
						return -1;
					}
					for (int lf2 = 0; lf2 < cells[c2].n_facets(); ++lf2)
					{
						if (c == adjacent[c2][lf2])
						{
							return first_cell_facet[c2] + lf2;
						}
					}
					assert(false);
					return -1;
				};

				std::vector<int> cell_facet_to_facet(first_cell_facet.back(), -1);

				// Creates 1 hex or polyhedral element for each cell of the input mesh
				int facet_counter = 0;
				bool is_hex = true;
				mesh_.cell_hex.reserve(cells.size());
				mesh_.cell_kernel.reserve(cells.size());
				std::vector<uint32_t> fs, face_vs, input_vs;
				std::vector<uint8_t> fs_flag;
				for (int c = 0; c < (int)cells.size(); ++c)
				{
					const bool hex = (cells[c].type == StandardCellType::HEX);
					mesh_.cell_hex.push_back(hex);
					is_hex = is_hex && hex;

					int nf = cells[c].n_facets();
					fs.assign(nf, 0);
					fs_flag.clear();

					for (int lf = 0; lf < nf; ++lf)
					{
						int cf = first_cell_facet[c] + lf;
						int cf2 = opposite_cell_facet(c, lf);
						if (cf2 < 0 || cell_facet_to_facet[cf2] < 0)
						{
							face_vs.resize(cells[c].n_facet_vertices(lf));
							for (int lv = 0; lv < cells[c].n_facet_vertices(lf); ++lv)
							{
								face_vs[lv] = cells[c].facet_vertex(lf, lv);
							}
							mesh_.face_vs.push_back(face_vs);
							fs_flag.push_back(0);
							fs[lf] = facet_counter;
							cell_facet_to_facet[cf] = facet_counter;
							++facet_counter;
						}
						else
						{
							fs[lf] = cell_facet_to_facet[cf2];
							fs_flag.push_back(1);
						}
					}
					mesh_.cell_fs.push_back(fs);
					mesh_.cell_fs_flag.push_back(fs_flag);
					mesh_.cell_vs.push_back_filled(0, 0); // built by prepare_mesh

					const int n_cell_vertices = cells[c].descriptor().n_vertices;
					input_vs.resize(n_cell_vertices);
					for (int lv = 0; lv < n_cell_vertices; ++lv)
						input_vs[lv] = cells[c].vertices[lv];
					mesh_.cell_input_vs.push_back(input_vs);

					// Compute a point in the kernel (assumes the barycenter is ok)
					Eigen::RowVector3d p(0, 0, 0);
					for (int v : input_vs)
					{
						p += mesh_.points.col(v).transpose();
					}
					p /= input_vs.size();
					mesh_.cell_kernel.push_back({{p[0], p[1], p[2]}});
				}
				mesh_.type = is_hex ? MeshType::HEX : (all_tets ? MeshType::TET : MeshType::HYB);
			}

			Navigation3D::prepare_mesh(mesh_);
			// if (is_simplicial()) {
			// 	MeshProcessing3D::orient_volume_mesh(mesh_);
			// }
			compute_elements_tag();
			return true;
		}

		bool CMesh3D::build_topology(const MeshData &data)
		{
			const Eigen::MatrixXd &V = data.vertices;
			const Eigen::MatrixXi &F = data.elements;
			edge_nodes_.clear();
			face_nodes_.clear();
			cell_nodes_.clear();

			if (data.has_polyhedral_topology())
			{
				mesh_ = Mesh3DStorage();
				mesh_.points = V.transpose();

				for (int i = 0; i < data.faces.size(); ++i)
					mesh_.face_vs.push_back(data.faces[i]);

				bool all_hex = true;
				std::vector<uint32_t> input_vs, vs;
				std::vector<uint8_t> fs_flag;
				for (int i = 0; i < F.rows(); ++i)
				{
					mesh_.cell_fs.push_back(data.cell_faces[i]);
					fs_flag.clear();
					for (const int o : data.cell_face_orientations[i])
						fs_flag.push_back(o != 0);
					mesh_.cell_fs_flag.push_back(fs_flag);
					mesh_.cell_hex.push_back(data.cell_is_hex[i]);
					all_hex &= data.cell_is_hex[i];
					input_vs.clear();
					if (data.elements_are_ordered)
						for (int j = 0; j < F.cols() && F(i, j) >= 0; ++j)
							input_vs.push_back(F(i, j));
					mesh_.cell_input_vs.push_back(input_vs);
					vs.clear();
					for (const int face : data.cell_faces[i])
						vs.insert(vs.end(), mesh_.face_vs[face].begin(), mesh_.face_vs[face].end());
					std::sort(vs.begin(), vs.end());
					vs.erase(std::unique(vs.begin(), vs.end()), vs.end());
					mesh_.cell_vs.push_back(vs);
					assert(data.cell_kernel_points.cols() == 3);
					mesh_.cell_kernel.push_back({{data.cell_kernel_points(i, 0), data.cell_kernel_points(i, 1), data.cell_kernel_points(i, 2)}});
				}
				mesh_.type = all_hex ? MeshType::HEX : MeshType::HYB;
				Navigation3D::prepare_mesh(mesh_);
				compute_elements_tag();
				return true;
			}
			assert(F.cols() == 4 || F.cols() == 5 || F.cols() == 6 || F.cols() == 8);

			static const std::vector<int> permute_tet = {0, 1, 2, 3};
			static const std::vector<int> permute_pyramid = {0, 1, 2, 3, 4};
			static const std::vector<int> permute_prism = {0, 1, 2, 3, 4, 5};
			// MeshData uses the msh hex vertex ordering, the cell descriptors use
			// geogram's (see CellConnectivity.hpp).
			static const std::vector<int> permute_hex = {1, 0, 2, 3, 5, 4, 6, 7};

			std::vector<StandardCell> cells;
			cells.reserve(F.rows());
			auto add_cell = [&](int c, StandardCellType t, int nv, const std::vector<int> &perm) {
				StandardCell cell(t);
				for (int lv = 0; lv < nv; ++lv)
				{
					int vi = F(c, perm[lv]);
					assert(vi >= 0 && vi < V.rows());
					cell.vertices[lv] = vi;
				}
				cells.push_back(cell);
			};

			for (int c = 0; c < F.rows(); ++c)
			{
				int nV;
				for (nV = 0; nV < F.cols(); ++nV)
				{
					if (F(c, nV) == -1)
						break;
				}

#ifndef NDEBUG
				for (int k = nV; k < F.cols(); ++k)
				{
					assert(F(c, k) == -1);
				}
#endif
				if (nV == 4)
					add_cell(c, StandardCellType::TET, 4, permute_tet);
				else if (nV == 5)
					add_cell(c, StandardCellType::PYRAMID, 5, permute_pyramid);
				else if (nV == 6)
					add_cell(c, StandardCellType::PRISM, 6, permute_prism);
				else if (nV == 8)
					add_cell(c, StandardCellType::HEX, 8, permute_hex);
				else
					log_and_throw_error(fmt::format("Invalid number of vertices {}", nV));
			}

			if (!build_storage_from_cells(V, cells))
				return false;

			// cell_vertex() (and therefore to_mesh_data(), checkpoints and the
			// MSH writer) returns input_vs, which must be in the input (MeshData,
			// i.e. Gmsh) order. Hex cells were permuted to the descriptor order
			// above; restore their input order (the kernel point was already
			// computed from the permuted order).
			for (int c = 0; c < cells.size(); ++c)
			{
				if (cells[c].type != StandardCellType::HEX)
					continue;
				for (int lv = 0; lv < 8; ++lv)
					mesh_.cell_input_vs[c][lv] = F(c, lv);
			}
			return true;
		}

		void CMesh3D::attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes)
		{
			edge_nodes_.clear();
			face_nodes_.clear();
			cell_nodes_.clear();

			edge_nodes_.resize(n_edges());
			face_nodes_.resize(n_faces());
			cell_nodes_.resize(n_cells());

			orders_.setOnes(n_cells(), 1);

			const auto attach_p2 = [&](const Navigation3D::Index &index, const std::vector<int> &nodes_ids) {
				auto &n = edge_nodes_[index.edge];

				if (n.nodes.size() > 0)
					return;

				n.v1 = index.vertex;
				n.v2 = switch_vertex(index).vertex;

				const int n_v1 = index.vertex;
				const int n_v2 = switch_vertex(index).vertex;

				int node_index = 0;

				if ((n_v1 == nodes_ids[0] && n_v2 == nodes_ids[1]) || (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[1]))
					node_index = 4;
				else if ((n_v1 == nodes_ids[1] && n_v2 == nodes_ids[2]) || (n_v2 == nodes_ids[1] && n_v1 == nodes_ids[2]))
					node_index = 5;
				else if ((n_v1 == nodes_ids[2] && n_v2 == nodes_ids[3]) || (n_v2 == nodes_ids[2] && n_v1 == nodes_ids[3]))
					node_index = 8;

				else if ((n_v1 == nodes_ids[0] && n_v2 == nodes_ids[3]) || (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[3]))
					node_index = 7;
				else if ((n_v1 == nodes_ids[0] && n_v2 == nodes_ids[2]) || (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[2]))
					node_index = 6;
				else
					node_index = 9;

				n.nodes.resize(1, 3);
				n.nodes << V(nodes_ids[node_index], 0), V(nodes_ids[node_index], 1), V(nodes_ids[node_index], 2);
				n.nodes_ids.push_back(nodes_ids[node_index]);
			};

			const auto attach_p3 = [&](const Navigation3D::Index &index, const std::vector<int> &nodes_ids) {
				auto &n = edge_nodes_[index.edge];

				if (n.nodes.size() > 0)
					return;

				n.v1 = index.vertex;
				n.v2 = switch_vertex(index).vertex;

				const int n_v1 = index.vertex;
				const int n_v2 = switch_vertex(index).vertex;

				int node_index1 = 0;
				int node_index2 = 0;
				if (n_v1 == nodes_ids[0] && n_v2 == nodes_ids[1])
				{
					node_index1 = 4;
					node_index2 = 5;
				}
				else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[1])
				{
					node_index1 = 5;
					node_index2 = 4;
				}
				else if (n_v1 == nodes_ids[1] && n_v2 == nodes_ids[2])
				{
					node_index1 = 6;
					node_index2 = 7;
				}
				else if (n_v2 == nodes_ids[1] && n_v1 == nodes_ids[2])
				{
					node_index1 = 7;
					node_index2 = 6;
				}
				else if (n_v1 == nodes_ids[2] && n_v2 == nodes_ids[3])
				{
					node_index1 = 13;
					node_index2 = 12;
				}
				else if (n_v2 == nodes_ids[2] && n_v1 == nodes_ids[3])
				{
					node_index1 = 12;
					node_index2 = 13;
				}

				else if (n_v1 == nodes_ids[0] && n_v2 == nodes_ids[3])
				{
					node_index1 = 11;
					node_index2 = 10;
				}
				else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[3])
				{
					node_index1 = 10;
					node_index2 = 11;
				}
				else if (n_v1 == nodes_ids[0] && n_v2 == nodes_ids[2])
				{
					node_index1 = 9;
					node_index2 = 8;
				}
				else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[2])
				{
					node_index1 = 8;
					node_index2 = 9;
				}

				else if (n_v2 == nodes_ids[1] && n_v1 == nodes_ids[3])
				{
					node_index1 = 14;
					node_index2 = 15;
				}
				else
				{
					node_index1 = 15;
					node_index2 = 14;
				}

				n.nodes.resize(2, 3);
				n.nodes.row(0) << V(nodes_ids[node_index1], 0), V(nodes_ids[node_index1], 1), V(nodes_ids[node_index1], 2);
				n.nodes.row(1) << V(nodes_ids[node_index2], 0), V(nodes_ids[node_index2], 1), V(nodes_ids[node_index2], 2);
				n.nodes_ids.push_back(nodes_ids[node_index1]);
				n.nodes_ids.push_back(nodes_ids[node_index2]);
			};

			const auto attach_p3_face = [&](const Navigation3D::Index &index, const std::vector<int> &nodes_ids, int id) {
				auto &n = face_nodes_[index.face];
				if (n.nodes.size() <= 0)
				{
					n.v1 = face_vertex(index.face, 0);
					n.v2 = face_vertex(index.face, 1);
					n.v3 = face_vertex(index.face, 2);
					n.nodes.resize(1, 3);
					n.nodes << V(nodes_ids[id], 0), V(nodes_ids[id], 1), V(nodes_ids[id], 2);
					n.nodes_ids.push_back(nodes_ids[id]);
				}
			};

			const auto attach_p4 = [&](const Navigation3D::Index &index, const std::vector<int> &nodes_ids) {
				auto &n = edge_nodes_[index.edge];

				if (n.nodes.size() > 0)
					return;

				n.v1 = index.vertex;
				n.v2 = switch_vertex(index).vertex;

				const int n_v1 = index.vertex;
				const int n_v2 = switch_vertex(index).vertex;

				int node_index1 = 0;
				int node_index2 = 0;
				int node_index3 = 0;
				if (n_v1 == nodes_ids[0] && n_v2 == nodes_ids[1])
				{
					node_index1 = 4;
					node_index2 = 5;
					node_index3 = 6;
				}
				else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[1])
				{
					node_index1 = 6;
					node_index2 = 5;
					node_index3 = 4;
				}

				else if (n_v1 == nodes_ids[1] && n_v2 == nodes_ids[2])
				{
					node_index1 = 7;
					node_index2 = 8;
					node_index3 = 9;
				}
				else if (n_v2 == nodes_ids[1] && n_v1 == nodes_ids[2])
				{
					node_index1 = 9;
					node_index2 = 8;
					node_index3 = 7;
				}

				else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[2])
				{
					node_index1 = 10;
					node_index2 = 11;
					node_index3 = 12;
				}
				else if (n_v1 == nodes_ids[0] && n_v2 == nodes_ids[2])
				{
					node_index1 = 12;
					node_index2 = 11;
					node_index3 = 10;
				}

				else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[3])
				{
					node_index1 = 13;
					node_index2 = 14;
					node_index3 = 15;
				}
				else if (n_v1 == nodes_ids[0] && n_v2 == nodes_ids[3])
				{
					node_index1 = 15;
					node_index2 = 14;
					node_index3 = 13;
				}

				else if (n_v2 == nodes_ids[2] && n_v1 == nodes_ids[3])
				{
					node_index1 = 16;
					node_index2 = 17;
					node_index3 = 18;
				}
				else if (n_v1 == nodes_ids[2] && n_v2 == nodes_ids[3])
				{
					node_index1 = 18;
					node_index2 = 17;
					node_index3 = 16;
				}

				else if (n_v2 == nodes_ids[1] && n_v1 == nodes_ids[3])
				{
					node_index1 = 19;
					node_index2 = 20;
					node_index3 = 21;
				}
				else if (n_v1 == nodes_ids[1] && n_v2 == nodes_ids[3])
				{
					node_index1 = 21;
					node_index2 = 20;
					node_index3 = 19;
				}
				else
				{
					assert(false);
				}

				n.nodes.resize(3, 3);
				n.nodes.row(0) << V(nodes_ids[node_index1], 0), V(nodes_ids[node_index1], 1), V(nodes_ids[node_index1], 2);
				n.nodes.row(1) << V(nodes_ids[node_index2], 0), V(nodes_ids[node_index2], 1), V(nodes_ids[node_index2], 2);
				n.nodes.row(2) << V(nodes_ids[node_index3], 0), V(nodes_ids[node_index3], 1), V(nodes_ids[node_index3], 2);
				n.nodes_ids.push_back(nodes_ids[node_index1]);
				n.nodes_ids.push_back(nodes_ids[node_index2]);
				n.nodes_ids.push_back(nodes_ids[node_index3]);
			};

			const auto attach_p4_face = [&](const Navigation3D::Index &index, const std::vector<int> &nodes_ids) {
				auto &n = face_nodes_[index.face];
				if (n.nodes.size() <= 0)
				{
					n.v1 = face_vertex(index.face, 0);
					n.v2 = face_vertex(index.face, 1);
					n.v3 = face_vertex(index.face, 2);

					std::array<int, 3> vid = {{n.v1, n.v2, n.v3}};
					std::sort(vid.begin(), vid.end());

					std::array<int, 3> c1 = {{nodes_ids[0], nodes_ids[1], nodes_ids[2]}}; // 22
					std::array<int, 3> c2 = {{nodes_ids[0], nodes_ids[1], nodes_ids[3]}}; // 25
					std::array<int, 3> c3 = {{nodes_ids[0], nodes_ids[2], nodes_ids[3]}}; // 28
					std::array<int, 3> c4 = {{nodes_ids[1], nodes_ids[2], nodes_ids[3]}}; // 31

					std::sort(c1.begin(), c1.end());
					std::sort(c2.begin(), c2.end());
					std::sort(c3.begin(), c3.end());
					std::sort(c4.begin(), c4.end());

					int id = 0;
					int index0 = 0;
					int index1 = 1;
					int index2 = 2;
					if (vid == c1)
					{
						id = 22;
						n.v1 = nodes_ids[0];
						n.v2 = nodes_ids[1];
						n.v3 = nodes_ids[2];

						index0 = 0;
						index1 = 2;
						index2 = 1; // ok
					}
					else if (vid == c2)
					{
						id = 25;
						n.v1 = nodes_ids[0];
						n.v2 = nodes_ids[1];
						n.v3 = nodes_ids[3];

						index0 = 0;
						index1 = 1;
						index2 = 2; // ok
					}
					else if (vid == c3)
					{
						id = 28;
						n.v1 = nodes_ids[0];
						n.v2 = nodes_ids[2];
						n.v3 = nodes_ids[3];

						index0 = 0;
						index1 = 2;
						index2 = 1; // ok
					}
					else if (vid == c4)
					{
						id = 31;
						n.v1 = nodes_ids[1];
						n.v2 = nodes_ids[2];
						n.v3 = nodes_ids[3];

						index0 = 1;
						index1 = 2;
						index2 = 0; // ok
					}
					else
					{
						// the face nees to be one of the 4 above
						assert(false);
					}

					n.nodes.resize(3, 3);
					assert(id + index0 < nodes_ids.size());
					assert(id + index1 < nodes_ids.size());
					assert(id + index2 < nodes_ids.size());
					n.nodes.row(0) << V(nodes_ids[id + index0], 0), V(nodes_ids[id + index0], 1), V(nodes_ids[id + index0], 2);
					n.nodes.row(1) << V(nodes_ids[id + index1], 0), V(nodes_ids[id + index1], 1), V(nodes_ids[id + index1], 2);
					n.nodes.row(2) << V(nodes_ids[id + index2], 0), V(nodes_ids[id + index2], 1), V(nodes_ids[id + index2], 2);
					n.nodes_ids.push_back(nodes_ids[id + index0]);
					n.nodes_ids.push_back(nodes_ids[id + index1]);
					n.nodes_ids.push_back(nodes_ids[id + index2]);
				}
			};

			const auto attach_p4_cell = [&](const Navigation3D::Index &index, const std::vector<int> &nodes_ids) {
				auto &n = cell_nodes_[index.element];
				assert(nodes_ids.size() == 35);
				assert(n.nodes.size() == 0);

				if (n.nodes.size() <= 0)
				{
					n.v1 = cell_vertex(index.element, 0);
					n.v2 = cell_vertex(index.element, 1);
					n.v3 = cell_vertex(index.element, 2);
					n.v4 = cell_vertex(index.element, 3);
					n.nodes.resize(1, 3);

					n.nodes << V(nodes_ids[34], 0), V(nodes_ids[34], 1), V(nodes_ids[34], 2);
					n.nodes_ids.push_back(nodes_ids[34]);
				}
			};

			assert(nodes.size() == n_cells());

			for (int c = 0; c < n_cells(); ++c)
			{
				auto index = get_index_from_element(c);

				const auto &nodes_ids = nodes[c];

				if (nodes_ids.size() == 4)
				{
					orders_(c) = 1;
					continue;
				}
				// P2
				else if (nodes_ids.size() == 10)
				{
					orders_(c) = 2;

					for (int le = 0; le < 3; ++le)
					{
						attach_p2(index, nodes_ids);
						index = next_around_face(index);
					}

					index = switch_vertex(switch_edge(switch_face(index)));
					attach_p2(index, nodes_ids);

					index = switch_edge(index);
					attach_p2(index, nodes_ids);

					index = switch_edge(switch_face(index));
					attach_p2(index, nodes_ids);
				}
				// P3
				else if (nodes_ids.size() == 20)
				{
					orders_(c) = 3;

					for (int le = 0; le < 3; ++le)
					{
						attach_p3(index, nodes_ids);
						index = next_around_face(index);
					}

					{
						index = switch_vertex(switch_edge(switch_face(index)));
						attach_p3(index, nodes_ids);

						index = switch_edge(index);
						attach_p3(index, nodes_ids);

						index = switch_edge(switch_face(index));
						attach_p3(index, nodes_ids);
					}

					{
						index = get_index_from_element(c);
						std::array<int, 4> indices;

						{
							std::array<int, 3> f16 = {{nodes_ids[0], nodes_ids[1], nodes_ids[2]}};
							std::array<int, 3> f17 = {{nodes_ids[3], nodes_ids[1], nodes_ids[0]}};
							std::array<int, 3> f18 = {{nodes_ids[0], nodes_ids[2], nodes_ids[3]}};
							std::array<int, 3> f19 = {{nodes_ids[1], nodes_ids[2], nodes_ids[3]}};
							std::sort(f16.begin(), f16.end());
							std::sort(f17.begin(), f17.end());
							std::sort(f18.begin(), f18.end());
							std::sort(f19.begin(), f19.end());

							auto tmp = index;
							std::array<int, 3> f0 = {{face_vertex(tmp.face, 0), face_vertex(tmp.face, 1), face_vertex(tmp.face, 2)}};
							tmp = switch_face(index);
							std::array<int, 3> f1 = {{face_vertex(tmp.face, 0), face_vertex(tmp.face, 1), face_vertex(tmp.face, 2)}};
							tmp = switch_face(next_around_face(index));
							std::array<int, 3> f2 = {{face_vertex(tmp.face, 0), face_vertex(tmp.face, 1), face_vertex(tmp.face, 2)}};
							tmp = switch_face(next_around_face(next_around_face(index)));
							std::array<int, 3> f3 = {{face_vertex(tmp.face, 0), face_vertex(tmp.face, 1), face_vertex(tmp.face, 2)}};

							std::sort(f0.begin(), f0.end());
							std::sort(f1.begin(), f1.end());
							std::sort(f2.begin(), f2.end());
							std::sort(f3.begin(), f3.end());

							const std::array<std::array<int, 3>, 4> faces = {{f0, f1, f2, f3}};
							const std::array<std::array<int, 3>, 4> nodes = {{f16, f17, f18, f19}};
							for (int i = 0; i < 4; ++i)
							{
								const auto &f = faces[i];
								bool found = false;
								for (int j = 0; j < 4; ++j)
								{
									if (nodes[j] == f)
									{
										indices[i] = j + 16;
										found = true;
										break;
									}
								}

								assert(found);
							}
						}

						attach_p3_face(index, nodes_ids, indices[0]);
						attach_p3_face(switch_face(index), nodes_ids, indices[1]);
						attach_p3_face(switch_face(next_around_face(index)), nodes_ids, indices[2]);
						attach_p3_face(switch_face(next_around_face(next_around_face(index))), nodes_ids, indices[3]);
					}
				}
				// P4
				else if (nodes_ids.size() == 35)
				{
					orders_(c) = 4;
					for (int le = 0; le < 3; ++le)
					{
						attach_p4(index, nodes_ids);
						index = next_around_face(index);
					}

					{
						index = switch_vertex(switch_edge(switch_face(index)));
						attach_p4(index, nodes_ids);

						index = switch_edge(index);
						attach_p4(index, nodes_ids);

						index = switch_edge(switch_face(index));
						attach_p4(index, nodes_ids);
					}

					{
						index = get_index_from_element(c);

						attach_p4_face(index, nodes_ids);
						attach_p4_face(switch_face(index), nodes_ids);
						attach_p4_face(switch_face(next_around_face(index)), nodes_ids);
						attach_p4_face(switch_face(next_around_face(next_around_face(index))), nodes_ids);
					}

					attach_p4_cell(get_index_from_element(c), nodes_ids);
				}
				// unsupported
				else
				{
					log_and_throw_error("Unsupported higher-order tetrahedron with {} nodes (supported: 4, 10, 20, 35).", nodes_ids.size());
				}
			}
		}

		void CMesh3D::bounding_box(RowVectorNd &min, RowVectorNd &max) const
		{
			const auto &V = mesh_.points;
			min = V.rowwise().minCoeff().transpose();
			max = V.rowwise().maxCoeff().transpose();
		}

		void CMesh3D::normalize()
		{
			RowVectorNd minV, maxV;
			bounding_box(minV, maxV);
			auto &V = mesh_.points;
			const auto shift = V.rowwise().minCoeff().eval();
			const double scaling = 1.0 / (V.rowwise().maxCoeff() - V.rowwise().minCoeff()).maxCoeff();
			V = (V.colwise() - shift) * scaling;

			for (int i = 0; i < n_cells(); ++i)
			{
				for (int d = 0; d < 3; ++d)
				{
					auto val = mesh_.cell_kernel[i][d];
					mesh_.cell_kernel[i][d] = (val - shift(d)) * scaling;
				}
			}

			for (auto &n : edge_nodes_)
			{
				if (n.nodes.size() > 0)
					n.nodes = (n.nodes.rowwise() - shift.transpose()) * scaling;
			}
			for (auto &n : face_nodes_)
			{
				if (n.nodes.size() > 0)
					n.nodes = (n.nodes.rowwise() - shift.transpose()) * scaling;
			}
			for (auto &n : cell_nodes_)
			{
				if (n.nodes.size() > 0)
					n.nodes = (n.nodes.rowwise() - shift.transpose()) * scaling;
			}
			transform_higher_order_data(
				Eigen::Matrix3d::Identity() * scaling,
				-shift * scaling);

			logger().debug("-- bbox before normalization:");
			logger().debug("   min   : {}", minV);
			logger().debug("   max   : {}", maxV);
			logger().debug("   extent: {}", maxV - minV);
			bounding_box(minV, maxV);
			logger().debug("-- bbox after normalization:");
			logger().debug("   min   : {}", minV);
			logger().debug("   max   : {}", maxV);
			logger().debug("   extent: {}", maxV - minV);

			// V.row(1) /= 100.;

			// for(int i = 0; i < n_cells(); ++i)
			// {
			// 	mesh_.cell_kernel[i][1] /= 100.;
			// }

			Eigen::MatrixXd p0, p1, p;
			get_edges(p0, p1);
			p = p0 - p1;
			logger().debug("-- edge length after normalization:");
			logger().debug("   min: ", p.rowwise().norm().minCoeff());
			logger().debug("   max: ", p.rowwise().norm().maxCoeff());
			logger().debug("   avg: ", p.rowwise().norm().mean());
		}

		// void CMesh3D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const
		// {
		// 	ranges.clear();

		// 	std::vector<Eigen::MatrixXi> local_tris(mesh_.n_cells());
		// 	std::vector<Eigen::MatrixXd> local_pts(mesh_.n_cells());
		// 	Eigen::MatrixXi tets;

		// 	int total_tris = 0;
		// 	int total_pts = 0;

		// 	ranges.push_back(0);

		// 	Eigen::MatrixXd face_barys;
		// 	face_barycenters(face_barys);

		// 	Eigen::MatrixXd cell_barys;
		// 	cell_barycenters(cell_barys);

		// 	for (std::size_t e = 0; e < mesh_.n_cells(); ++e)
		// 	{
		// 		const Element &el = mesh_.elements[e];

		// 		const int n_vertices = el.vs.size();
		// 		const int n_faces = el.fs.size();

		// 		Eigen::MatrixXd local_pt(n_vertices + n_faces, 3);

		// 		std::map<int, int> global_to_local;

		// 		for (int i = 0; i < n_vertices; ++i)
		// 		{
		// 			const int global_index = el.vs[i];
		// 			local_pt.row(i) = mesh_.points.col(global_index).transpose();
		// 			global_to_local[global_index] = i;
		// 		}

		// 		int n_local_faces = 0;
		// 		for (int i = 0; i < n_faces; ++i)
		// 		{
		// 			const Face &f = mesh_.faces[el.fs[i]];
		// 			n_local_faces += f.vs.size();

		// 			local_pt.row(n_vertices + i) = face_barys.row(f.id); // node_from_face(f.id);
		// 		}

		// 		Eigen::MatrixXi local_faces(n_local_faces, 3);

		// 		int face_index = 0;
		// 		for (int i = 0; i < n_faces; ++i)
		// 		{
		// 			const Face &f = mesh_.faces[el.fs[i]];
		// 			const int n_face_vertices = f.vs.size();

		// 			const Eigen::RowVector3d e0 = (point(f.vs[0]) - local_pt.row(n_vertices + i));
		// 			const Eigen::RowVector3d e1 = (point(f.vs[1]) - local_pt.row(n_vertices + i));
		// 			const Eigen::RowVector3d normal = e0.cross(e1);
		// 			// const Eigen::RowVector3d check_dir = (node_from_element(e)-p);
		// 			const Eigen::RowVector3d check_dir = (cell_barys.row(e) - point(f.vs[1]));

		// 			const bool reverse_order = normal.dot(check_dir) > 0;

		// 			for (int j = 0; j < n_face_vertices; ++j)
		// 			{
		// 				const int jp = (j + 1) % n_face_vertices;
		// 				if (reverse_order)
		// 				{
		// 					local_faces(face_index, 0) = global_to_local[f.vs[jp]];
		// 					local_faces(face_index, 1) = global_to_local[f.vs[j]];
		// 				}
		// 				else
		// 				{
		// 					local_faces(face_index, 0) = global_to_local[f.vs[j]];
		// 					local_faces(face_index, 1) = global_to_local[f.vs[jp]];
		// 				}
		// 				local_faces(face_index, 2) = n_vertices + i;

		// 				++face_index;
		// 			}
		// 		}

		// 		local_pts[e] = local_pt;
		// 		local_tris[e] = local_faces;

		// 		total_tris += local_tris[e].rows();
		// 		total_pts += local_pts[e].rows();

		// 		ranges.push_back(total_tris);

		// 		assert(local_pts[e].rows() == local_pt.rows());
		// 	}

		// 	tris.resize(total_tris, 3);
		// 	pts.resize(total_pts, 3);

		// 	int tri_index = 0;
		// 	int pts_index = 0;
		// 	for (std::size_t i = 0; i < local_tris.size(); ++i)
		// 	{
		// 		tris.block(tri_index, 0, local_tris[i].rows(), local_tris[i].cols()) = local_tris[i].array() + pts_index;
		// 		tri_index += local_tris[i].rows();

		// 		pts.block(pts_index, 0, local_pts[i].rows(), local_pts[i].cols()) = local_pts[i];
		// 		pts_index += local_pts[i].rows();
		// 	}
		// }

		bool CMesh3D::is_boundary_element(const int element_global_id) const
		{
			const auto &fs = mesh_.cell_fs[element_global_id];

			for (auto f_id : fs)
			{
				if (is_boundary_face(f_id))
					return true;
			}

			const auto &vs = mesh_.cell_vs[element_global_id];

			for (auto v_id : vs)
			{
				if (is_boundary_vertex(v_id))
					return true;
			}

			return false;
		}

		void CMesh3D::compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker)
		{
			boundary_ids_.resize(n_faces());

			for (int f = 0; f < n_faces(); ++f)
			{
				const bool is_boundary = is_boundary_face(f);
				std::vector<int> vs(n_face_vertices(f));
				for (int vid = 0; vid < vs.size(); ++vid)
					vs[vid] = face_vertex(f, vid);

				const auto p = face_barycenter(f);

				std::sort(vs.begin(), vs.end());
				boundary_ids_[f] = marker(f, vs, p, is_boundary);
			}
		}

		void CMesh3D::compute_body_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &)> &marker)
		{
			body_ids_.resize(n_elements());
			std::fill(body_ids_.begin(), body_ids_.end(), -1);

			for (int e = 0; e < n_elements(); ++e)
			{
				const auto bary = cell_barycenter(e);
				body_ids_[e] = marker(e, element_vertices(e), bary);
			}
		}

		void CMesh3D::apply_affine_transformation(const MatrixNd &A, const VectorNd &b)
		{
			Mesh::apply_affine_transformation(A, b);
			for (auto &kernel : mesh_.cell_kernel)
			{
				VectorNd k(3);
				k << kernel[0], kernel[1], kernel[2];
				k = A * k + b;
				for (int d = 0; d < 3; ++d)
					kernel[d] = k[d];
			}
		}

		void CMesh3D::set_point(const int global_index, const RowVectorNd &p)
		{
			mesh_.points.col(global_index) = p.transpose();
		}

		RowVectorNd CMesh3D::point(const int global_index) const
		{
			RowVectorNd pt = mesh_.points.col(global_index).transpose();
			return pt;
		}

		RowVectorNd CMesh3D::kernel(const int c) const
		{
			RowVectorNd pt(3);
			pt << mesh_.cell_kernel[c][0], mesh_.cell_kernel[c][1], mesh_.cell_kernel[c][2];
			return pt;
		}

		void CMesh3D::compute_elements_tag()
		{
			std::vector<ElementType> &ele_tag = elements_tag_;
			ele_tag.clear();

			ele_tag.resize(mesh_.n_cells());
			for (auto &t : ele_tag)
				t = ElementType::REGULAR_INTERIOR_CUBE;

			// boundary flags
			std::vector<bool> bv_flag(mesh_.n_vertices(), false), be_flag(mesh_.n_edges(), false), bf_flag(mesh_.n_faces(), false);
			for (int f = 0; f < mesh_.n_faces(); ++f)
				if (mesh_.face_boundary[f])
					bf_flag[f] = true;
				else
				{
					for (auto nhid : mesh_.face_hs[f])
						if (!mesh_.cell_hex[nhid])
							bf_flag[f] = true;
				}
			for (uint32_t i = 0; i < mesh_.n_faces(); ++i)
				if (bf_flag[i])
					for (uint32_t j = 0; j < mesh_.face_vs[i].size(); ++j)
					{
						uint32_t eid = mesh_.face_es[i][j];
						be_flag[eid] = true;
						bv_flag[mesh_.face_vs[i][j]] = true;
					}

			for (int c = 0; c < mesh_.n_cells(); ++c)
			{
				const auto ele_vs = mesh_.cell_vs[c];
				const auto ele_es = mesh_.cell_es[c];
				if (mesh_.cell_hex[c])
				{
					bool attaching_non_hex = false, on_boundary = false;
					;
					for (auto vid : ele_vs)
					{
						for (auto eleid : mesh_.vertex_hs[vid])
							if (!mesh_.cell_hex[eleid])
							{
								attaching_non_hex = true;
								break;
							}
						if (mesh_.vertex_boundary[vid])
						{
							on_boundary = true;
							break;
						}
						if (on_boundary || attaching_non_hex)
							break;
					}
					if (attaching_non_hex)
					{
						ele_tag[c] = ElementType::INTERFACE_CUBE;
						continue;
					}

					if (on_boundary)
					{
						ele_tag[c] = ElementType::MULTI_SINGULAR_BOUNDARY_CUBE;
						// has no boundary edge--> singular
						bool boundary_edge = false, boundary_edge_singular = false, interior_edge_singular = false;
						int n_interior_edge_singular = 0;
						for (auto eid : ele_es)
						{
							int en = 0;
							if (be_flag[eid])
							{
								boundary_edge = true;
								for (auto nhid : mesh_.edge_hs[eid])
									if (mesh_.cell_hex[nhid])
										en++;
								if (en > 2)
									boundary_edge_singular = true;
							}
							else
							{
								for (auto nhid : mesh_.edge_hs[eid])
									if (mesh_.cell_hex[nhid])
										en++;
								if (en != 4)
								{
									interior_edge_singular = true;
									n_interior_edge_singular++;
								}
							}
						}
						if (!boundary_edge || boundary_edge_singular || n_interior_edge_singular > 1)
							continue;

						bool has_singular_v = false, has_iregular_v = false;
						int n_in_irregular_v = 0;
						for (auto vid : ele_vs)
						{
							int vn = 0;
							if (bv_flag[vid])
							{
								int nh = 0;
								for (auto nhid : mesh_.vertex_hs[vid])
									if (mesh_.cell_hex[nhid])
										nh++;
								if (nh > 4)
									has_iregular_v = true;
								continue; // not sure the conditions
							}
							else
							{
								if (mesh_.vertex_hs[vid].size() != 8)
									n_in_irregular_v++;
								int n_irregular_e = 0;
								for (auto eid : mesh_.vertex_es[vid])
								{
									if (mesh_.edge_hs[eid].size() != 4)
										n_irregular_e++;
								}
								if (n_irregular_e != 0 && n_irregular_e != 2)
								{
									has_singular_v = true;
									break;
								}
							}
						}
						int n_irregular_e = 0;
						for (auto eid : ele_es)
							if (!be_flag[eid] && mesh_.edge_hs[eid].size() != 4)
								n_irregular_e++;
						if (has_singular_v)
							continue;
						if (!has_singular_v)
						{
							if (n_irregular_e == 1)
							{
								ele_tag[c] = ElementType::SIMPLE_SINGULAR_BOUNDARY_CUBE;
							}
							else if (n_irregular_e == 0 && n_in_irregular_v == 0 && !has_iregular_v)
								ele_tag[c] = ElementType::REGULAR_BOUNDARY_CUBE;
							else
								continue;
						}
						continue;
					}

					// type 1
					bool has_irregular_v = false;
					for (auto vid : ele_vs)
						if (mesh_.vertex_hs[vid].size() != 8)
						{
							has_irregular_v = true;
							break;
						}
					if (!has_irregular_v)
					{
						ele_tag[c] = ElementType::REGULAR_INTERIOR_CUBE;
						continue;
					}
					// type 2
					bool has_singular_v = false;
					int n_irregular_v = 0;
					for (auto vid : ele_vs)
					{
						if (mesh_.vertex_hs[vid].size() != 8)
							n_irregular_v++;
						int n_irregular_e = 0;
						for (auto eid : mesh_.vertex_es[vid])
						{
							if (mesh_.edge_hs[eid].size() != 4)
								n_irregular_e++;
						}
						if (n_irregular_e != 0 && n_irregular_e != 2)
						{
							has_singular_v = true;
							break;
						}
					}
					if (!has_singular_v && n_irregular_v == 2)
					{
						ele_tag[c] = ElementType::SIMPLE_SINGULAR_INTERIOR_CUBE;
						continue;
					}

					ele_tag[c] = ElementType::MULTI_SINGULAR_INTERIOR_CUBE;
				}
				else
				{
					ele_tag[c] = ElementType::INTERIOR_POLYTOPE;
					for (auto fid : mesh_.cell_fs[c])
						if (mesh_.face_boundary[fid])
						{
							ele_tag[c] = ElementType::BOUNDARY_POLYTOPE;
							break;
						}
				}
			}

			// TODO correct?
			for (int c = 0; c < mesh_.n_cells(); ++c)
			{
				const int nv = mesh_.cell_vs.row_size(c);
				if (nv == 4)
					ele_tag[c] = ElementType::SIMPLEX;
				else if (nv == 5)
					ele_tag[c] = ElementType::PYRAMID;
				else if (nv == 6)
					ele_tag[c] = ElementType::PRISM;
			}
		}

		double CMesh3D::quad_area(const int gid) const
		{
			const int n_vertices = n_face_vertices(gid);
			assert(n_vertices == 4);

			const auto &vertices = mesh_.face_vs[gid];

			const auto v1 = point(vertices[0]);
			const auto v2 = point(vertices[1]);
			const auto v3 = point(vertices[2]);
			const auto v4 = point(vertices[3]);

			const Eigen::Vector3d e0 = (v2 - v1).transpose();
			const Eigen::Vector3d e1 = (v3 - v1).transpose();

			const Eigen::Vector3d e2 = (v2 - v4).transpose();
			const Eigen::Vector3d e3 = (v3 - v4).transpose();

			return e0.cross(e1).norm() / 2 + e2.cross(e3).norm() / 2;
		}

		RowVectorNd CMesh3D::edge_barycenter(const int e) const
		{
			const int v0 = mesh_.edge_vs[e][0];
			const int v1 = mesh_.edge_vs[e][1];
			return 0.5 * (point(v0) + point(v1));
		}

		RowVectorNd CMesh3D::face_barycenter(const int f) const
		{
			const int n_vertices = n_face_vertices(f);
			RowVectorNd bary(3);
			bary.setZero();

			const auto &vertices = mesh_.face_vs[f];
			for (int lv = 0; lv < n_vertices; ++lv)
			{
				bary += point(vertices[lv]);
			}
			return bary / n_vertices;
		}

		RowVectorNd CMesh3D::cell_barycenter(const int c) const
		{
			const int n_vertices = n_cell_vertices(c);
			RowVectorNd bary(3);
			bary.setZero();

			const auto &vertices = mesh_.cell_vs[c];
			for (int lv = 0; lv < n_vertices; ++lv)
			{
				bary += point(vertices[lv]);
			}
			return bary / n_vertices;
		}

		void CMesh3D::barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coord) const
		{
			assert(is_simplex(el_id));

			const auto indices = get_ordered_vertices_from_tet(el_id);

			const auto A = point(indices[0]);
			const auto B = point(indices[1]);
			const auto C = point(indices[2]);
			const auto D = point(indices[3]);

			igl::barycentric_coordinates(p, A, B, C, D, coord);
		}

		void CMesh3D::append(const Mesh &mesh)
		{
			assert(typeid(mesh) == typeid(CMesh3D));
			Mesh::append(mesh);

			const CMesh3D &mesh3d = dynamic_cast<const CMesh3D &>(mesh);
			mesh_.append(mesh3d.mesh_);

			Navigation3D::prepare_mesh(mesh_);
			compute_elements_tag();
		}

		std::unique_ptr<Mesh> CMesh3D::copy() const
		{
			return std::make_unique<CMesh3D>(*this);
		}

	} // namespace mesh
} // namespace polyfem
