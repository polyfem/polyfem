#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/writeMESH.h>

#include <geogram/mesh/mesh_io.h>
#include <fstream>

using namespace polyfem::utils;

namespace polyfem
{
	namespace mesh
	{
		int NCMesh3D::face_edge(const int f_id, const int le_id) const
		{
			const int v0 = faces[valid_to_all_face(f_id)].vertices(le_id);
			const int v1 = faces[valid_to_all_face(f_id)].vertices((le_id + 1) % 3);
			const int e_id = find_edge(v0, v1);
			return all_to_valid_edge(e_id);
		}

		void NCMesh3D::refine(const int n_refinement, const double t)
		{
			if (n_refinement <= 0)
				return;
			std::vector<bool> refine_mask(elements.size(), false);
			for (int i = 0; i < elements.size(); i++)
				if (elements[i].is_valid())
					refine_mask[i] = true;

			for (int i = 0; i < refine_mask.size(); i++)
				if (refine_mask[i])
					refine_element(i);

			refine(n_refinement - 1, t);
		}

		bool NCMesh3D::is_boundary_element(const int element_global_id) const
		{
			assert(index_prepared);
			for (int lv = 0; lv < n_cell_edges(element_global_id); lv++)
				if (is_boundary_vertex(cell_vertex(element_global_id, lv)))
					return true;

			return false;
		}

		bool NCMesh3D::build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
		{
			n_elements = 0;
			elements.clear();
			vertices.clear();
			edges.clear();
			midpointMap.clear();
			edgeMap.clear();
			faceMap.clear();
			refineHistory.clear();
			index_prepared = false;
			adj_prepared = false;

			vertices.reserve(V.rows());
			for (int i = 0; i < V.rows(); i++)
			{
				vertices.emplace_back(V.row(i));
			}
			for (int i = 0; i < F.rows(); i++)
			{
				add_element(F.row(i), -1);
			}

			prepare_mesh();

			return true;
		}

		void NCMesh3D::attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes)
		{
			for (int f = 0; f < n_cells(); ++f)
				if (nodes[f].size() != 4)
					throw std::runtime_error("NCMesh doesn't support high order mesh!");
		}

		void NCMesh3D::normalize()
		{
			polyfem::RowVectorNd min, max;
			bounding_box(min, max);

			auto extent = max - min;
			double scale = extent.maxCoeff();

			for (auto &v : vertices)
				v.pos = (v.pos - min.transpose()) / scale;
		}

		void NCMesh3D::compute_elements_tag()
		{
			elements_tag_.assign(n_cells(), ElementType::SIMPLEX);
		}
		void NCMesh3D::update_elements_tag()
		{
			elements_tag_.assign(n_cells(), ElementType::SIMPLEX);
		}
		double NCMesh3D::edge_length(const int gid) const
		{
			const int v1 = edge_vertex(gid, 0);
			const int v2 = edge_vertex(gid, 1);

			return (point(v1) - point(v2)).norm();
		}

		RowVectorNd NCMesh3D::point(const int global_index) const
		{
			return vertices[valid_to_all_vertex(global_index)].pos.transpose();
		}

		void NCMesh3D::set_point(const int global_index, const RowVectorNd &p)
		{
			vertices[valid_to_all_vertex(global_index)].pos = p;
		}

		RowVectorNd NCMesh3D::edge_barycenter(const int e) const
		{
			const int v1 = edge_vertex(e, 0);
			const int v2 = edge_vertex(e, 1);

			return 0.5 * (point(v1) + point(v2));
		}
		RowVectorNd NCMesh3D::face_barycenter(const int f) const
		{
			const int v1 = face_vertex(f, 0);
			const int v2 = face_vertex(f, 1);
			const int v3 = face_vertex(f, 2);

			return (point(v1) + point(v2) + point(v3)) / 3.;
		}
		RowVectorNd NCMesh3D::cell_barycenter(const int c) const
		{
			const int v1 = face_vertex(c, 0);
			const int v2 = face_vertex(c, 1);
			const int v3 = face_vertex(c, 2);

			return (point(v1) + point(v2) + point(v3)) / 3.;
		}

		void NCMesh3D::bounding_box(RowVectorNd &min, RowVectorNd &max) const
		{
			min = vertices[0].pos;
			max = vertices[0].pos;

			for (const auto &v : vertices)
			{
				for (int d = 0; d < 3; d++)
				{
					if (v.pos[d] > max[d])
						max[d] = v.pos[d];
					if (v.pos[d] < min[d])
						min[d] = v.pos[d];
				}
			}
		}

		void NCMesh3D::compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker)
		{
			boundary_ids_.resize(n_faces());
			std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

			for (int f = 0; f < n_faces(); ++f)
			{
				const bool is_boundary = is_boundary_face(f);
				std::vector<int> vs(n_face_vertices(f));
				const auto p = face_barycenter(f);

				for (int vid = 0; vid < vs.size(); ++vid)
					vs[vid] = face_vertex(f, vid);

				std::sort(vs.begin(), vs.end());
				boundary_ids_[f] = marker(f, vs, p, is_boundary);

				faces[valid_to_all_face(f)].boundary_id = boundary_ids_[f];
			}
		}

		void NCMesh3D::compute_body_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &)> &marker)
		{
			body_ids_.resize(n_cells());
			std::fill(body_ids_.begin(), body_ids_.end(), -1);

			for (int e = 0; e < n_cells(); ++e)
			{
				const auto bary = cell_barycenter(e);
				body_ids_[e] = marker(e, element_vertices(e), bary);
				elements[valid_to_all_elem(e)].body_id = body_ids_[e];
			}
		}
		void NCMesh3D::set_boundary_ids(const std::vector<int> &boundary_ids)
		{
			assert(boundary_ids.size() == n_faces());
			for (int i = 0; i < boundary_ids.size(); i++)
			{
				faces[valid_to_all_face(i)].boundary_id = boundary_ids[i];
			}
		}
		void NCMesh3D::set_body_ids(const std::vector<int> &body_ids)
		{
			assert(body_ids.size() == n_cells());
			for (int i = 0; i < body_ids.size(); i++)
			{
				elements[valid_to_all_elem(i)].body_id = body_ids[i];
			}
		}

		// void NCMesh3D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const
		// {
		// 	ranges.clear();

		// 	std::vector<Eigen::MatrixXi> local_tris(n_cells());
		// 	std::vector<Eigen::MatrixXd> local_pts(n_cells());
		// 	Eigen::MatrixXi tets;

		// 	int total_tris = 0;
		// 	int total_pts = 0;

		// 	ranges.push_back(0);

		// 	Eigen::MatrixXd face_barys;
		// 	face_barycenters(face_barys);

		// 	Eigen::MatrixXd cell_barys;
		// 	cell_barycenters(cell_barys);

		// 	for (std::size_t e = 0; e < n_cells(); ++e)
		// 	{
		// 		const int n_vertices = n_cell_vertices(e);
		// 		const int n_faces = n_cell_faces(e);

		// 		Eigen::MatrixXd local_pt(n_vertices + n_faces, 3);

		// 		std::map<int, int> global_to_local;

		// 		for (int i = 0; i < n_vertices; ++i)
		// 		{
		// 			const int global_index = cell_vertex(e, i);
		// 			local_pt.row(i) = point(global_index);
		// 			global_to_local[global_index] = i;
		// 		}

		// 		int n_local_faces = 0;
		// 		for (int i = 0; i < n_faces; ++i)
		// 		{
		// 			const int f_id = cell_face(e, i);
		// 			n_local_faces += n_face_vertices(f_id);

		// 			local_pt.row(n_vertices + i) = face_barys.row(f_id);
		// 		}

		// 		Eigen::MatrixXi local_faces(n_local_faces, 3);

		// 		int face_index = 0;
		// 		for (int i = 0; i < n_faces; ++i)
		// 		{
		// 			const int f_id = cell_face(e, i);

		// 			const Eigen::RowVector3d e0 = (point(face_vertex(f_id, 0)) - local_pt.row(n_vertices + i));
		// 			const Eigen::RowVector3d e1 = (point(face_vertex(f_id, 1)) - local_pt.row(n_vertices + i));
		// 			const Eigen::RowVector3d normal = e0.cross(e1);
		// 			// const Eigen::RowVector3d check_dir = (node_from_element(e)-p);
		// 			const Eigen::RowVector3d check_dir = (cell_barys.row(e) - point(face_vertex(f_id, 1)));

		// 			const bool reverse_order = normal.dot(check_dir) > 0;

		// 			for (int j = 0; j < n_face_vertices(f_id); ++j)
		// 			{
		// 				const int jp = (j + 1) % n_face_vertices(f_id);
		// 				if (reverse_order)
		// 				{
		// 					local_faces(face_index, 0) = global_to_local[face_vertex(f_id, jp)];
		// 					local_faces(face_index, 1) = global_to_local[face_vertex(f_id, j)];
		// 				}
		// 				else
		// 				{
		// 					local_faces(face_index, 0) = global_to_local[face_vertex(f_id, j)];
		// 					local_faces(face_index, 1) = global_to_local[face_vertex(f_id, jp)];
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

		RowVectorNd NCMesh3D::kernel(const int cell_id) const
		{
			assert(false);
			return RowVectorNd();
		}
		Navigation3D::Index NCMesh3D::get_index_from_element(int hi, int lf, int lv) const
		{
			Navigation3D::Index idx;

			idx.element = hi;

			idx.element_patch = lf;
			idx.face = cell_face(hi, lf);

			if (lv >= 4)
				lv = lv % 4;
			idx.face_corner = lv;
			idx.vertex = face_vertex(idx.face, idx.face_corner);

			idx.edge = face_edge(idx.face, idx.face_corner);

			return idx;
		}
		Navigation3D::Index NCMesh3D::get_index_from_element(int hi) const
		{
			Navigation3D::Index idx;

			idx.element = hi;
			idx.element_patch = 0;
			idx.face = cell_face(hi, 0);
			idx.face_corner = 0;
			idx.vertex = face_vertex(idx.face, idx.face_corner);
			idx.edge = face_edge(idx.face, 0);

			return idx;
		}

		Navigation3D::Index NCMesh3D::get_index_from_element_edge(int hi, int v0, int v1) const
		{
			Navigation3D::Index idx;

			idx.element = hi;
			idx.vertex = v0;
			idx.edge = all_to_valid_edge(find_edge(valid_to_all_vertex(v0), valid_to_all_vertex(v1)));

			for (int i = 0; i < 4; i++)
			{
				const int f_id = cell_face(idx.element, i);
				for (int j = 0; j < 3; j++)
				{
					const int e_id = face_edge(f_id, j);
					if (idx.edge == e_id)
					{
						idx.element_patch = i;
						idx.face = f_id;

						for (int d = 0; d < n_face_vertices(f_id); d++)
							if (face_vertex(f_id, d) == idx.vertex)
								idx.face_corner = d;

						assert(switch_vertex(idx).vertex == v1);

						return idx;
					}
				}
			}

			assert(false);
			return idx;
		}
		Navigation3D::Index NCMesh3D::get_index_from_element_face(int hi, int v0, int v1, int v2) const
		{
			Navigation3D::Index idx;

			idx.element = hi;
			idx.vertex = v0;
			idx.face = all_to_valid_face(find_face(valid_to_all_vertex(v0), valid_to_all_vertex(v1), valid_to_all_vertex(v2)));

			for (int d = 0; d < n_face_vertices(idx.face); d++)
				if (face_vertex(idx.face, d) == idx.vertex)
					idx.face_corner = d;

			int v0_ = v0, v1_ = v1;

			// let v0_ < v1_
			if (v0_ > v1_)
				std::swap(v0_, v1_);

			for (int i = 0; i < 4; i++)
			{
				const int fid = cell_face(idx.element, i);
				if (fid != idx.face)
					continue;

				idx.element_patch = i;

				for (int j = 0; j < 3; j++)
				{
					const int eid = face_edge(fid, j);
					const int ev0 = edge_vertex(eid, 0);
					const int ev1 = edge_vertex(eid, 1);

					if ((ev0 == v0_ && ev1 == v1_) || (ev0 == v1_ && ev1 == v0_))
					{
						idx.edge = eid;

						assert(switch_vertex(idx).vertex == v1);
						assert(switch_vertex(switch_edge(idx)).vertex == v2);

						return idx;
					}
				}
			}

			assert(false);
			return idx;
		}

		std::vector<uint32_t> NCMesh3D::vertex_neighs(const int v_gid) const
		{
			std::vector<uint32_t> hs;
			for (auto h : vertices[valid_to_all_vertex(v_gid)].elem_list)
				hs.push_back(all_to_valid_elem(h));

			return hs;
		}
		std::vector<uint32_t> NCMesh3D::edge_neighs(const int e_gid) const
		{
			std::vector<uint32_t> hs;
			for (auto h : edges[valid_to_all_edge(e_gid)].elem_list)
				hs.push_back(all_to_valid_elem(h));

			return hs;
		}

		Navigation3D::Index NCMesh3D::switch_vertex(Navigation3D::Index idx) const
		{
			if (idx.vertex == edge_vertex(idx.edge, 0))
				idx.vertex = edge_vertex(idx.edge, 1);
			else
				idx.vertex = edge_vertex(idx.edge, 0);

			for (int d = 0; d < n_face_vertices(idx.face); d++)
				if (face_vertex(idx.face, d) == idx.vertex)
					idx.face_corner = d;

			return idx;
		}
		Navigation3D::Index NCMesh3D::switch_edge(Navigation3D::Index idx) const
		{
			if (idx.edge == face_edge(idx.face, idx.face_corner))
				idx.edge = face_edge(idx.face, (idx.face_corner + 2) % 3);
			else
				idx.edge = face_edge(idx.face, idx.face_corner);
			return idx;
		}
		Navigation3D::Index NCMesh3D::switch_face(Navigation3D::Index idx) const
		{
			for (int i = 0; i < 4; i++)
			{
				const int fid = cell_face(idx.element, i);
				if (fid == idx.face)
					continue;
				if (idx.edge == face_edge(fid, 0) || idx.edge == face_edge(fid, 1) || idx.edge == face_edge(fid, 2))
				{
					idx.face = fid;
					idx.element_patch = i;

					for (int d = 0; d < n_face_vertices(fid); d++)
						if (face_vertex(fid, d) == idx.vertex)
							idx.face_corner = d;

					break;
				}
			}
			return idx;
		}
		Navigation3D::Index NCMesh3D::switch_element(Navigation3D::Index idx) const
		{
			const auto &face = faces[valid_to_all_face(idx.face)];
			if (face.n_elem() != 2)
			{
				idx.element = -1;
				return idx;
			}
			if (all_to_valid_elem(face.get_element()) == idx.element)
				idx.element = all_to_valid_elem(face.find_opposite_element(face.get_element()));
			else
				idx.element = all_to_valid_elem(face.get_element());

			for (int f = 0; f < n_cell_faces(idx.element); f++)
				if (cell_face(idx.element, f) == idx.face)
					idx.element_patch = f;
			return idx;
		}

		Navigation3D::Index NCMesh3D::next_around_edge(Navigation3D::Index idx) const
		{
			return switch_element(switch_face(idx));
		}
		Navigation3D::Index NCMesh3D::next_around_face(Navigation3D::Index idx) const
		{
			return switch_edge(switch_vertex(idx));
		}

		void NCMesh3D::get_vertex_elements_neighs(const int v_id, std::vector<int> &ids) const
		{
			ids.clear();
			for (auto h : vertices[valid_to_all_vertex(v_id)].elem_list)
				ids.push_back(all_to_valid_elem(h));
		}
		void NCMesh3D::get_edge_elements_neighs(const int e_id, std::vector<int> &ids) const
		{
			ids.clear();
			for (auto h : edges[valid_to_all_edge(e_id)].elem_list)
				ids.push_back(all_to_valid_elem(h));
		}
		void NCMesh3D::get_face_elements_neighs(const int f_id, std::vector<int> &ids) const
		{
			ids.clear();
			for (auto h : faces[valid_to_all_face(f_id)].elem_list)
				ids.push_back(all_to_valid_elem(h));
		}

		void NCMesh3D::refine_element(int id_full)
		{
			assert(elements[id_full].is_valid());
			if (elements[id_full].is_not_valid())
				throw std::runtime_error("Cannot refine an invalid element!");

			const auto v = elements[id_full].vertices;
			elements[id_full].is_refined = true;
			n_elements--;

			for (int f = 0; f < elements[id_full].faces.size(); f++)
				faces[elements[id_full].faces(f)].remove_element(id_full);

			for (int e = 0; e < elements[id_full].edges.size(); e++)
				edges[elements[id_full].edges(e)].remove_element(id_full);

			for (int i = 0; i < v.size(); i++)
				vertices[v(i)].remove_element(id_full);

			if (elements[id_full].children(0) >= 0)
			{
				for (int c = 0; c < elements[id_full].children.size(); c++)
				{
					const int child_id = elements[id_full].children(c);
					auto &elem = elements[child_id];
					elem.is_ghost = false;
					n_elements++;

					for (int f = 0; f < elem.faces.size(); f++)
						faces[elem.faces(f)].add_element(child_id);

					for (int e = 0; e < elem.edges.size(); e++)
						edges[elem.edges(e)].add_element(child_id);

					for (int v = 0; v < elem.vertices.size(); v++)
						vertices[elem.vertices(v)].add_element(child_id);
				}
			}
			else
			{
				// create mid-points if not exist
				const int v1 = v[0];
				const int v2 = v[1];
				const int v3 = v[2];
				const int v4 = v[3];
				const int v5 = get_vertex(Eigen::Vector2i(v1, v2));
				const int v8 = get_vertex(Eigen::Vector2i(v3, v2));
				const int v6 = get_vertex(Eigen::Vector2i(v1, v3));
				const int v7 = get_vertex(Eigen::Vector2i(v1, v4));
				const int v9 = get_vertex(Eigen::Vector2i(v4, v2));
				const int v10 = get_vertex(Eigen::Vector2i(v3, v4));

				// inherite line singularity flag from parent edge
				for (int i = 0; i < v.size(); i++)
					for (int j = 0; j < i; j++)
					{
						int mid_id = find_vertex(v[i], v[j]);
						int edge_id = find_edge(v[i], v[j]);
						int edge1 = get_edge(v[i], mid_id);
						int edge2 = get_edge(v[j], mid_id);
						edges[edge1].boundary_id = edges[edge_id].boundary_id;
						edges[edge2].boundary_id = edges[edge_id].boundary_id;
					}

				for (int i = 0; i < v.size(); i++)
					for (int j = 0; j < i; j++)
						for (int k = 0; k < j; k++)
						{
							int fid = find_face(v[i], v[j], v[k]);
							int vij = find_vertex(v[i], v[j]);
							int vjk = find_vertex(v[j], v[k]);
							int vik = find_vertex(v[i], v[k]);
							int facei = get_face(v[i], vij, vik);
							int facej = get_face(v[j], vjk, vij);
							int facek = get_face(v[k], vjk, vik);
							int facem = get_face(vij, vjk, vik);
							faces[facei].boundary_id = faces[fid].boundary_id;
							faces[facej].boundary_id = faces[fid].boundary_id;
							faces[facek].boundary_id = faces[fid].boundary_id;
							faces[facem].boundary_id = faces[fid].boundary_id;
						}

				// create children
				elements[id_full].children(0) = elements.size();
				add_element(Eigen::Vector4i(v1, v5, v6, v7), id_full);
				elements[id_full].children(1) = elements.size();
				add_element(Eigen::Vector4i(v5, v2, v8, v9), id_full);
				elements[id_full].children(2) = elements.size();
				add_element(Eigen::Vector4i(v6, v8, v3, v10), id_full);
				elements[id_full].children(3) = elements.size();
				add_element(Eigen::Vector4i(v7, v9, v10, v4), id_full);
				elements[id_full].children(4) = elements.size();
				add_element(Eigen::Vector4i(v5, v6, v7, v9), id_full);
				elements[id_full].children(5) = elements.size();
				add_element(Eigen::Vector4i(v5, v9, v8, v6), id_full);
				elements[id_full].children(6) = elements.size();
				add_element(Eigen::Vector4i(v6, v7, v9, v10), id_full);
				elements[id_full].children(7) = elements.size();
				add_element(Eigen::Vector4i(v6, v10, v9, v8), id_full);
			}

			refineHistory.push_back(id_full);
		}
		void NCMesh3D::refine_elements(const std::vector<int> &ids)
		{
			std::vector<int> full_ids(ids.size());
			for (int i = 0; i < ids.size(); i++)
				full_ids[i] = valid_to_all_elem(ids[i]);

			for (int i : full_ids)
				refine_element(i);
		}

		void NCMesh3D::coarsen_element(int id_full)
		{
			const int parent_id = elements[id_full].parent;
			auto &parent = elements[parent_id];

			for (int i = 0; i < parent.children.size(); i++)
				if (elements[parent.children(i)].is_not_valid())
					throw std::runtime_error("Invalid siblings in coarsening!");

			// remove elements
			for (int c = 0; c < parent.children.size(); c++)
			{
				auto &elem = elements[parent.children(c)];
				elem.is_ghost = true;
				n_elements--;

				for (int f = 0; f < elem.faces.size(); f++)
					faces[elem.faces(f)].remove_element(parent.children(c));

				for (int e = 0; e < elem.edges.size(); e++)
					edges[elem.edges(e)].remove_element(parent.children(c));

				for (int v = 0; v < elem.vertices.size(); v++)
					vertices[elem.vertices(v)].remove_element(parent.children(c));
			}

			// add element
			parent.is_refined = false;
			n_elements++;

			for (int f = 0; f < parent.faces.size(); f++)
				faces[parent.faces(f)].add_element(parent_id);

			for (int e = 0; e < parent.edges.size(); e++)
				edges[parent.edges(e)].add_element(parent_id);

			for (int v = 0; v < parent.vertices.size(); v++)
				vertices[parent.vertices(v)].add_element(parent_id);

			refineHistory.push_back(parent_id);
		}

		void NCMesh3D::mark_boundary()
		{
			for (auto &face : faces)
			{
				if (face.n_elem() == 1)
					face.isboundary = true;
				else
					face.isboundary = false;
			}

			for (auto &face : faces)
			{
				if (face.leader >= 0)
				{
					face.isboundary = false;
					faces[face.leader].isboundary = false;
				}
			}

			for (auto &vert : vertices)
				vert.isboundary = false;

			for (auto &edge : edges)
				edge.isboundary = false;

			for (auto &face : faces)
			{
				if (face.isboundary && face.n_elem())
				{
					for (int j = 0; j < 3; j++)
						vertices[face.vertices(j)].isboundary = true;

					for (int j = 0; j < 3; j++)
						for (int i = 0; i < j; i++)
							edges[find_edge(face.vertices(i), face.vertices(j))].isboundary = true;
				}
			}

			// for (int v = 0; v < n_vertices(); v++)
			//     std::cout << v << ": " << point(v) << "\n";

			// for (int f = 0; f < n_faces(); f++)
			// {
			//     for (int v = 0; v < n_face_vertices(f); v++)
			//         std::cout << face_vertex(f, v) << ", ";
			//     std::cout << faces[valid_to_all_face(f)].isboundary << std::endl;
			// }
		}

		void NCMesh3D::build_index_mapping()
		{
			all_to_valid_elemMap.assign(elements.size(), -1);
			valid_to_all_elemMap.resize(n_elements);

			for (int i = 0, e = 0; i < elements.size(); i++)
			{
				if (elements[i].is_not_valid())
					continue;
				all_to_valid_elemMap[i] = e;
				valid_to_all_elemMap[e] = i;
				e++;
			}

			const int n_verts = n_vertices();

			all_to_valid_vertexMap.assign(vertices.size(), -1);
			valid_to_all_vertexMap.resize(n_verts);

			for (int i = 0, j = 0; i < vertices.size(); i++)
			{
				if (vertices[i].n_elem() == 0)
					continue;
				all_to_valid_vertexMap[i] = j;
				valid_to_all_vertexMap[j] = i;
				j++;
			}

			all_to_valid_edgeMap.assign(edges.size(), -1);
			valid_to_all_edgeMap.resize(n_edges());

			for (int i = 0, j = 0; i < edges.size(); i++)
			{
				if (edges[i].n_elem() == 0)
					continue;
				all_to_valid_edgeMap[i] = j;
				valid_to_all_edgeMap[j] = i;
				j++;
			}

			all_to_valid_faceMap.assign(faces.size(), -1);
			valid_to_all_faceMap.resize(n_faces());

			for (int i = 0, j = 0; i < faces.size(); i++)
			{
				if (faces[i].n_elem() == 0)
					continue;
				all_to_valid_faceMap[i] = j;
				valid_to_all_faceMap[j] = i;
				j++;
			}
			index_prepared = true;
		}

		void NCMesh3D::append(const Mesh &mesh)
		{
			assert(typeid(mesh) == typeid(NCMesh3D));
			Mesh::append(mesh);

			const NCMesh3D &mesh3d = dynamic_cast<const NCMesh3D &>(mesh);

			const int n_v = n_vertices();
			const int n_f = n_cells();

			vertices.reserve(n_v + mesh3d.n_vertices());
			for (int i = 0; i < mesh3d.n_vertices(); i++)
			{
				vertices.emplace_back(mesh3d.vertices[i].pos);
			}
			for (int i = 0; i < mesh3d.n_cells(); i++)
			{
				Eigen::Vector4i cell = mesh3d.elements[i].vertices;
				cell = cell.array() + n_v;
				add_element(cell, -1);
			}

			prepare_mesh();
		}

		std::unique_ptr<Mesh> NCMesh3D::copy() const
		{
			return std::make_unique<NCMesh3D>(*this);
		}

		bool NCMesh3D::load(const std::string &path)
		{
			if (!StringUtils::endswith(path, ".HYBRID"))
			{
				GEO::Mesh M;
				GEO::mesh_load(path, M);
				return load(M);
			}
			return false;
		}
		bool NCMesh3D::load(const GEO::Mesh &M)
		{
			assert(M.vertices.dimension() == 3);

			Eigen::MatrixXd V(M.vertices.nb(), 2);
			Eigen::MatrixXi C(M.cells.nb(), 4);

			for (int v = 0; v < V.rows(); v++)
				V.row(v) << M.vertices.point(v)[0], M.vertices.point(v)[1], M.vertices.point(v)[2];

			for (int c = 0; c < C.rows(); c++)
			{
				if (M.cells.type(c) != GEO::MESH_TET)
					throw std::runtime_error("NCMesh3D only supports tet mesh!");
				for (int i = 0; i < C.cols(); i++)
					C(c, i) = M.cells.vertex(c, i);
			}

			n_elements = 0;
			vertices.reserve(V.rows());
			for (int i = 0; i < V.rows(); i++)
			{
				vertices.emplace_back(V.row(i));
			}
			for (int i = 0; i < C.rows(); i++)
			{
				add_element(C.row(i), -1);
			}

			prepare_mesh();

			return true;
		}

		int NCMesh3D::find_vertex(Eigen::Vector2i v) const
		{
			std::sort(v.data(), v.data() + v.size());
			auto search = midpointMap.find(v);
			if (search != midpointMap.end())
				return search->second;
			else
				return -1;
		}
		int NCMesh3D::get_vertex(Eigen::Vector2i v)
		{
			std::sort(v.data(), v.data() + v.size());
			int id = find_vertex(v);
			if (id < 0)
			{
				Eigen::VectorXd v_mid = (vertices[v[0]].pos + vertices[v[1]].pos) / 2.;
				id = vertices.size();
				vertices.emplace_back(v_mid);
				midpointMap.emplace(v, id);
			}
			return id;
		}
		int NCMesh3D::find_edge(Eigen::Vector2i v) const
		{
			std::sort(v.data(), v.data() + v.size());
			auto search = edgeMap.find(v);
			if (search != edgeMap.end())
				return search->second;
			else
				return -1;
		}
		int NCMesh3D::get_edge(Eigen::Vector2i v)
		{
			std::sort(v.data(), v.data() + v.size());
			int id = find_edge(v);
			if (id < 0)
			{
				edges.emplace_back(v);
				id = edges.size() - 1;
				edgeMap.emplace(v, id);
			}
			return id;
		}
		int NCMesh3D::find_face(Eigen::Vector3i v) const
		{
			std::sort(v.data(), v.data() + v.size());
			auto search = faceMap.find(v);
			if (search != faceMap.end())
				return search->second;
			else
				return -1;
		}
		int NCMesh3D::get_face(Eigen::Vector3i v)
		{
			std::sort(v.data(), v.data() + v.size());
			int id = find_face(v);
			if (id < 0)
			{
				faces.emplace_back(v);
				id = faces.size() - 1;
				faceMap.emplace(v, id);
			}
			return id;
		}
		void NCMesh3D::traverse_edge(Eigen::Vector2i v, double p1, double p2, int depth, std::vector<follower_edge> &list) const
		{
			int v_mid = find_vertex(v);
			std::vector<follower_edge> list1, list2;
			if (v_mid >= 0)
			{
				double p_mid = (p1 + p2) / 2;
				traverse_edge(Eigen::Vector2i(v[0], v_mid), p1, p_mid, depth + 1, list1);
				list.insert(
					list.end(),
					std::make_move_iterator(list1.begin()),
					std::make_move_iterator(list1.end()));
				traverse_edge(Eigen::Vector2i(v_mid, v[1]), p_mid, p2, depth + 1, list2);
				list.insert(
					list.end(),
					std::make_move_iterator(list2.begin()),
					std::make_move_iterator(list2.end()));
			}
			if (depth > 0)
			{
				int follower_id = find_edge(v);
				if (follower_id >= 0 && edges[follower_id].n_elem() > 0)
					list.emplace_back(follower_id, p1, p2);
			}
		}
		void NCMesh3D::build_edge_follower_chain()
		{
			for (auto &edge : edges)
			{
				edge.leader = -1;
				edge.followers.clear();
				edge.weights.setConstant(-1);
			}

			for (int e_id = 0; e_id < edges.size(); e_id++)
			{
				auto &edge = edges[e_id];
				if (edge.n_elem() == 0)
					continue;
				std::vector<follower_edge> followers;
				traverse_edge(edge.vertices, 0, 1, 0, followers);
				for (auto &s : followers)
				{
					if (edges[s.id].leader >= 0 && std::abs(edges[s.id].weights(1) - edges[s.id].weights(0)) < std::abs(s.p2 - s.p1))
						continue;
					edge.followers.push_back(s.id);
					edges[s.id].leader = e_id;
					edges[s.id].weights << s.p1, s.p2;
				}
			}

			// In 3d, it's possible for one edge to have both leader and follower edges, but we don't care this case.
			for (auto &edge : edges)
				if (edge.leader >= 0 && edge.followers.size())
					edge.followers.clear();
		}
		void NCMesh3D::traverse_face(int v1, int v2, int v3, Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Vector2d p3, int depth, std::vector<follower_face> &face_list, std::vector<int> &edge_list) const
		{
			int v12 = find_vertex(Eigen::Vector2i(v1, v2));
			int v23 = find_vertex(Eigen::Vector2i(v3, v2));
			int v31 = find_vertex(Eigen::Vector2i(v1, v3));
			std::vector<follower_face> list1, list2, list3, list4;
			std::vector<int> list1_, list2_, list3_, list4_;
			if (depth > 0)
			{
				edge_list.push_back(find_edge(v1, v2));
				edge_list.push_back(find_edge(v3, v2));
				edge_list.push_back(find_edge(v1, v3));
			}
			if (v12 >= 0 && v23 >= 0 && v31 >= 0)
			{
				auto p12 = (p1 + p2) / 2, p23 = (p2 + p3) / 2, p31 = (p1 + p3) / 2;
				traverse_face(v1, v12, v31, p1, p12, p31, depth + 1, list1, list1_);
				traverse_face(v12, v2, v23, p12, p2, p23, depth + 1, list2, list2_);
				traverse_face(v31, v23, v3, p31, p23, p3, depth + 1, list3, list3_);
				traverse_face(v12, v23, v31, p12, p23, p31, depth + 1, list4, list4_);

				face_list.insert(face_list.end(), std::make_move_iterator(list1.begin()), std::make_move_iterator(list1.end()));
				face_list.insert(face_list.end(), std::make_move_iterator(list2.begin()), std::make_move_iterator(list2.end()));
				face_list.insert(face_list.end(), std::make_move_iterator(list3.begin()), std::make_move_iterator(list3.end()));
				face_list.insert(face_list.end(), std::make_move_iterator(list4.begin()), std::make_move_iterator(list4.end()));

				edge_list.insert(edge_list.end(), std::make_move_iterator(list1_.begin()), std::make_move_iterator(list1_.end()));
				edge_list.insert(edge_list.end(), std::make_move_iterator(list2_.begin()), std::make_move_iterator(list2_.end()));
				edge_list.insert(edge_list.end(), std::make_move_iterator(list3_.begin()), std::make_move_iterator(list3_.end()));
				edge_list.insert(edge_list.end(), std::make_move_iterator(list4_.begin()), std::make_move_iterator(list4_.end()));
			}
			if (depth > 0)
			{
				int follower_id = find_face(Eigen::Vector3i(v1, v2, v3));
				if (follower_id >= 0 && faces[follower_id].n_elem() > 0)
					face_list.emplace_back(follower_id, p1, p2, p3);
			}
		}
		void NCMesh3D::build_face_follower_chain()
		{
			Eigen::Matrix<int, 4, 3> fv;
			fv.row(0) << 0, 1, 2;
			fv.row(1) << 0, 1, 3;
			fv.row(2) << 1, 2, 3;
			fv.row(3) << 2, 0, 3;

			for (auto &face : faces)
			{
				face.leader = -1;
				face.followers.clear();
			}

			for (auto &edge : edges)
			{
				edge.leader_face = -1;
			}

			for (int f_id = 0; f_id < faces.size(); f_id++)
			{
				auto &face = faces[f_id];
				if (face.n_elem() == 0)
					continue;
				std::vector<follower_face> followers;
				std::vector<int> interior_edges;
				traverse_face(face.vertices(0), face.vertices(1), face.vertices(2), Eigen::Vector2d(0, 0), Eigen::Vector2d(1, 0), Eigen::Vector2d(0, 1), 0, followers, interior_edges); // order is important
				for (auto &s : followers)
				{
					faces[s.id].leader = f_id;
					face.followers.push_back(s.id);
				}
				for (int s : interior_edges)
					if (s >= 0 && edges[s].leader < 0 && edges[s].n_elem() > 0)
						edges[s].leader_face = f_id;
			}
		}
		void NCMesh3D::build_element_vertex_adjacency()
		{
			for (auto &vert : vertices)
			{
				vert.edge = -1;
				vert.face = -1;
			}

			for (auto &small_edge : edges)
			{
				// invalid edges
				if (small_edge.n_elem() == 0)
					continue;

				// not follower edges
				int large_edge = small_edge.leader;
				if (large_edge < 0)
					continue;
				assert(edges[large_edge].leader < 0);

				// follower edges
				for (int j = 0; j < 2; j++)
				{
					const int v_id = small_edge.vertices(j);
					// hanging nodes
					if (v_id != edges[large_edge].vertices(0) && v_id != edges[large_edge].vertices(1))
						vertices[v_id].edge = large_edge;
				}
			}

			for (auto &small_face : faces)
			{
				// invalid faces
				if (small_face.n_elem() == 0)
					continue;

				// not follower faces
				int large_face = small_face.leader;
				if (large_face < 0)
					continue;

				// follower faces
				for (int j = 0; j < 3; j++)
				{
					const int v_id = small_face.vertices(j);
					// hanging nodes
					if (v_id != faces[large_face].vertices(0) && v_id != faces[large_face].vertices(1) && v_id != faces[large_face].vertices(2))
						vertices[v_id].face = large_face;
				}
			}
		}
		std::array<int, 4> NCMesh3D::get_ordered_vertices_from_tet(const int element_index) const
		{
			return std::array<int, 4>({{cell_vertex(element_index, 0), cell_vertex(element_index, 1), cell_vertex(element_index, 2), cell_vertex(element_index, 3)}});
		}
		int NCMesh3D::add_element(Eigen::Vector4i v, int parent)
		{
			const int id = elements.size();
			const int level = (parent < 0) ? 0 : elements[parent].level + 1;

			Eigen::Vector3d e1 = vertices[v[1]].pos - vertices[v[0]].pos;
			Eigen::Vector3d e2 = vertices[v[2]].pos - vertices[v[0]].pos;
			Eigen::Vector3d e3 = vertices[v[3]].pos - vertices[v[0]].pos;
			if ((e1.cross(e2)).dot(e3) < 0)
				std::swap(v[2], v[3]);

			e2 = vertices[v[2]].pos - vertices[v[0]].pos;
			e3 = vertices[v[3]].pos - vertices[v[0]].pos;
			assert((e1.cross(e2)).dot(e3) > 0);

			elements.emplace_back(3, v, level, parent);

			if (parent >= 0)
				elements[id].body_id = elements[parent].body_id;

			// add faces if not exist
			const int face012 = get_face(v[0], v[1], v[2]);
			const int face013 = get_face(v[0], v[1], v[3]);
			const int face123 = get_face(v[1], v[2], v[3]);
			const int face203 = get_face(v[2], v[0], v[3]);

			faces[face012].add_element(id);
			faces[face013].add_element(id);
			faces[face123].add_element(id);
			faces[face203].add_element(id);

			elements[id].faces << face012, face013, face123, face203;

			// add edges if not exist
			const int edge01 = get_edge(v[0], v[1]);
			const int edge12 = get_edge(v[1], v[2]);
			const int edge20 = get_edge(v[2], v[0]);
			const int edge03 = get_edge(v[0], v[3]);
			const int edge13 = get_edge(v[1], v[3]);
			const int edge23 = get_edge(v[2], v[3]);

			edges[edge01].add_element(id);
			edges[edge12].add_element(id);
			edges[edge20].add_element(id);
			edges[edge03].add_element(id);
			edges[edge13].add_element(id);
			edges[edge23].add_element(id);

			elements[id].edges << edge01, edge12, edge20, edge03, edge13, edge23;

			n_elements++;

			for (int i = 0; i < v.size(); i++)
				vertices[v[i]].add_element(id);

			return id;
		}
	} // namespace mesh
} // namespace polyfem
