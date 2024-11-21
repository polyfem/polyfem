#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/writeOBJ.h>

#include <polyfem/mesh/MeshUtils.hpp>

namespace polyfem
{
	namespace mesh
	{
		bool NCMesh2D::is_boundary_element(const int element_global_id) const
		{
			assert(index_prepared);
			for (int le = 0; le < n_face_vertices(element_global_id); le++)
				if (is_boundary_edge(face_edge(element_global_id, le)))
					return true;

			return false;
		}

		void NCMesh2D::refine(const int n_refinement, const double t)
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

		bool NCMesh2D::build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
		{
			GEO::Mesh mesh_;
			mesh_.clear(false, false);
			to_geogram_mesh(V, F, mesh_);
			orient_normals_2d(mesh_);

			n_elements = 0;
			elements.clear();
			vertices.clear();
			edges.clear();
			midpointMap.clear();
			edgeMap.clear();
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

		void NCMesh2D::attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes)
		{
			for (int f = 0; f < n_faces(); ++f)
				if (nodes[f].size() != 3)
					throw std::runtime_error("NCMesh doesn't support high order mesh!");
		}
		std::pair<RowVectorNd, int> NCMesh2D::edge_node(const Navigation::Index &index, const int n_new_nodes, const int i) const
		{
			const auto v1 = point(index.vertex);
			const auto v2 = point(switch_vertex(index).vertex);

			const double t = i / (n_new_nodes + 1.0);

			return std::make_pair((1 - t) * v1 + t * v2, -1);
		}
		std::pair<RowVectorNd, int> NCMesh2D::face_node(const Navigation::Index &index, const int n_new_nodes, const int i, const int j) const
		{
			const auto v1 = point(index.vertex);
			const auto v2 = point(switch_vertex(index).vertex);
			const auto v3 = point(switch_vertex(switch_edge(index)).vertex);

			const double b2 = i / (n_new_nodes + 2.0);
			const double b3 = j / (n_new_nodes + 2.0);
			const double b1 = 1 - b3 - b2;
			assert(b3 < 1);
			assert(b3 > 0);

			return std::make_pair(b1 * v1 + b2 * v2 + b3 * v3, -1);
		}

		int NCMesh2D::find_vertex(Eigen::Vector2i v) const
		{
			std::sort(v.data(), v.data() + v.size());
			auto search = midpointMap.find(v);
			if (search != midpointMap.end())
				return search->second;
			else
				return -1;
		}

		int NCMesh2D::get_vertex(Eigen::Vector2i v)
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

		int NCMesh2D::find_edge(Eigen::Vector2i v) const
		{
			std::sort(v.data(), v.data() + v.size());
			auto search = edgeMap.find(v);
			if (search != edgeMap.end())
				return search->second;
			else
				return -1;
		}

		int NCMesh2D::get_edge(Eigen::Vector2i v)
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

		int NCMesh2D::add_element(Eigen::Vector3i v, int parent)
		{
			const int id = elements.size();
			const int level = (parent < 0) ? 0 : elements[parent].level + 1;
			elements.emplace_back(2, v, level, parent);

			if (parent >= 0)
				elements[id].body_id = elements[parent].body_id;

			for (int i = 0; i < v.size(); i++)
				vertices[v(i)].n_elem++;

			// add edges if not exist
			int edge01 = get_edge(Eigen::Vector2i(v[0], v[1]));
			int edge12 = get_edge(Eigen::Vector2i(v[2], v[1]));
			int edge20 = get_edge(Eigen::Vector2i(v[0], v[2]));

			elements[id].edges << edge01, edge12, edge20;

			edges[edge01].add_element(id);
			edges[edge12].add_element(id);
			edges[edge20].add_element(id);

			n_elements++;
			index_prepared = false;
			adj_prepared = false;

			return id;
		}

		void NCMesh2D::refine_element(int id_full)
		{
			auto &elem = elements[id_full];
			if (elem.is_not_valid())
				throw std::runtime_error("Cannot refine an invalid element!");

			const auto v = elem.vertices;
			elem.is_refined = true;
			n_elements--;

			// remove the old element from edge reference
			for (int e = 0; e < 3; e++)
				edges[elem.edges(e)].remove_element(id_full);

			for (int i = 0; i < v.size(); i++)
				vertices[v(i)].n_elem--;

			if (elem.children(0) >= 0)
			{
				for (int c = 0; c < elem.children.size(); c++)
				{
					auto &child = elements[elem.children(c)];
					child.is_ghost = false;
					n_elements++;
					for (int le = 0; le < child.edges.size(); le++)
						edges[child.edges(le)].add_element(child.children(c));
					for (int i = 0; i < child.vertices.size(); i++)
						vertices[child.vertices(i)].n_elem++;
				}
			}
			else
			{
				// create mid-points if not exist
				const int v01 = get_vertex(Eigen::Vector2i(v[0], v[1]));
				const int v12 = get_vertex(Eigen::Vector2i(v[2], v[1]));
				const int v20 = get_vertex(Eigen::Vector2i(v[0], v[2]));

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

				// create and insert child elements
				elements[id_full].children(0) = elements.size();
				add_element(Eigen::Vector3i(v[0], v01, v20), id_full);
				elements[id_full].children(1) = elements.size();
				add_element(Eigen::Vector3i(v[1], v12, v01), id_full);
				elements[id_full].children(2) = elements.size();
				add_element(Eigen::Vector3i(v[2], v20, v12), id_full);
				elements[id_full].children(3) = elements.size();
				add_element(Eigen::Vector3i(v12, v20, v01), id_full);
			}

			refineHistory.push_back(id_full);

			index_prepared = false;
			adj_prepared = false;
		}

		void NCMesh2D::refine_elements(const std::vector<int> &ids)
		{
			std::vector<int> full_ids(ids.size());
			for (int i = 0; i < ids.size(); i++)
				full_ids[i] = valid_to_all_elem(ids[i]);

			for (int i : full_ids)
				refine_element(i);
		}

		void NCMesh2D::coarsen_element(int id_full)
		{
			const int parent_id = elements[id_full].parent;
			auto &parent = elements[parent_id];

			for (int i = 0; i < parent.children.size(); i++)
				if (elements[parent.children(i)].is_not_valid())
					throw std::runtime_error("Coarsen operation invalid!");

			// remove elements
			for (int i = 0; i < parent.children.size(); i++)
			{
				auto &elem = elements[parent.children(i)];
				elem.is_ghost = true;
				n_elements--;
				for (int le = 0; le < elem.edges.size(); le++)
					edges[elem.edges(le)].remove_element(parent.children(i));
				for (int v = 0; v < elem.vertices.size(); v++)
					vertices[elem.vertices(v)].n_elem--;
			}

			// add element
			parent.is_refined = false;
			n_elements++;
			for (int le = 0; le < parent.edges.size(); le++)
				edges[parent.edges(le)].add_element(parent_id);
			for (int v = 0; v < parent.vertices.size(); v++)
				vertices[parent.vertices(v)].n_elem++;

			refineHistory.push_back(parent_id);

			index_prepared = false;
			adj_prepared = false;
		}

		int find(const Eigen::VectorXi &vec, int x)
		{
			for (int i = 0; i < vec.size(); i++)
			{
				if (x == vec[i])
					return i;
			}
			return -1;
		}

		void NCMesh2D::build_edge_follower_chain()
		{
			for (auto &edge : edges)
			{
				edge.leader = -1;
				edge.followers.clear();
				edge.weights.setConstant(-1);
			}

			Eigen::Vector2i v;
			std::vector<follower_edge> followers;
			for (int e_id = 0; e_id < elements.size(); e_id++)
			{
				const auto &element = elements[e_id];
				if (element.is_not_valid())
					continue;
				for (int edge_local = 0; edge_local < 3; edge_local++)
				{
					v << element.vertices[edge_local], element.vertices[(edge_local + 1) % 3]; // order is important here!
					int edge_global = element.edges[edge_local];
					assert(edge_global >= 0);
					traverse_edge(v, 0, 1, 0, followers);
					for (auto &s : followers)
					{
						edges[s.id].leader = edge_global;
						edges[edge_global].followers.push_back(s.id);
						edges[s.id].weights << s.p1, s.p2;
					}
					followers.clear();
				}
			}
		}

		void NCMesh2D::mark_boundary()
		{
			for (auto &edge : edges)
			{
				if (edge.n_elem() == 1)
					edge.isboundary = true;
				else
					edge.isboundary = false;
			}

			for (auto &edge : edges)
			{
				if (edge.leader >= 0)
				{
					edge.isboundary = false;
					edges[edge.leader].isboundary = false;
				}
			}

			for (auto &vert : vertices)
				vert.isboundary = false;

			for (auto &edge : edges)
			{
				if (edge.isboundary && edge.n_elem())
				{
					for (int j = 0; j < 2; j++)
						vertices[edge.vertices(j)].isboundary = true;
				}
			}
		}

		double line_weight(Eigen::Matrix<double, 2, 2> &e, Eigen::VectorXd &v)
		{
			assert(v.size() == 2);
			double w1 = (v(0) - e(0, 0)) / (e(1, 0) - e(0, 0));
			double w2 = (v(1) - e(0, 1)) / (e(1, 1) - e(0, 1));
			if (0 <= w1 && w1 <= 1)
				return w1;
			else
				return w2;
		}

		void NCMesh2D::build_element_vertex_adjacency()
		{
			for (auto &vert : vertices)
			{
				vert.edge = -1;
				vert.weight = -1;
			}

			Eigen::VectorXi vertexEdgeAdjacency;
			vertexEdgeAdjacency.setConstant(vertices.size(), 1, -1);

			for (int small_edge = 0; small_edge < edges.size(); small_edge++)
			{
				if (edges[small_edge].n_elem() == 0)
					continue;

				int large_edge = edges[small_edge].leader;
				if (large_edge < 0)
					continue;

				int large_elem = edges[large_edge].get_element();
				for (int j = 0; j < 2; j++)
				{
					int v_id = edges[small_edge].vertices(j);
					if (find(elements[large_elem].vertices, v_id) < 0) // or maybe 0 < weights(large_edge, v_id) < 1
						vertexEdgeAdjacency[v_id] = large_edge;
				}
			}

			for (auto &element : elements)
			{
				if (element.is_not_valid())
					continue;
				for (int v_local = 0; v_local < 3; v_local++)
				{
					int v_global = element.vertices[v_local];
					if (vertexEdgeAdjacency[v_global] < 0)
						continue;

					auto &large_edge = edges[vertexEdgeAdjacency[v_global]];
					auto &large_element = elements[large_edge.get_element()];
					vertices[v_global].edge = vertexEdgeAdjacency[v_global];

					int e_local = find(large_element.edges, vertices[v_global].edge);
					Eigen::Matrix<double, 2, 2> edge;
					edge.row(0) = vertices[large_element.vertices[e_local]].pos;
					edge.row(1) = vertices[large_element.vertices[(e_local + 1) % 3]].pos;
					vertices[v_global].weight = line_weight(edge, vertices[v_global].pos);
				}
			}
		}

		double NCMesh2D::element_weight_to_edge_weight(const int l, const Eigen::Vector2d &pos)
		{
			double w = -1;
			switch (l)
			{
			case 0:
				w = pos(0);
				assert(fabs(pos(1)) < 1e-12);
				break;
			case 1:
				w = pos(1);
				assert(fabs(pos(0) + pos(1) - 1) < 1e-12);
				break;
			case 2:
				w = 1 - pos(1);
				assert(fabs(pos(0)) < 1e-12);
				break;
			default:
				assert(false);
			}
			return w;
		}

		void NCMesh2D::build_index_mapping()
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
				if (vertices[i].n_elem == 0)
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
			index_prepared = true;
		}

		void NCMesh2D::append(const Mesh &mesh)
		{
			assert(typeid(mesh) == typeid(NCMesh2D));
			Mesh::append(mesh);

			const NCMesh2D &mesh2d = dynamic_cast<const NCMesh2D &>(mesh);

			const int n_v = n_vertices();
			const int n_f = n_faces();

			vertices.reserve(n_v + mesh2d.n_vertices());
			for (int i = 0; i < mesh2d.n_vertices(); i++)
			{
				vertices.emplace_back(mesh2d.vertices[i].pos);
			}
			for (int i = 0; i < mesh2d.n_faces(); i++)
			{
				Eigen::Vector3i face = mesh2d.elements[i].vertices;
				face = face.array() + n_v;
				add_element(face, -1);
			}

			prepare_mesh();
		}

		std::unique_ptr<Mesh> NCMesh2D::copy() const
		{
			return std::make_unique<NCMesh2D>(*this);
		}

		void NCMesh2D::traverse_edge(Eigen::Vector2i v, double p1, double p2, int depth, std::vector<follower_edge> &list) const
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

		bool NCMesh2D::load(const std::string &path)
		{
			assert(false);
			return false;
		}

		bool NCMesh2D::load(const GEO::Mesh &mesh)
		{
			GEO::Mesh mesh_;
			mesh_.clear(false, false);
			mesh_.copy(mesh);
			orient_normals_2d(mesh_);

			Eigen::MatrixXd V(mesh_.vertices.nb(), 2);
			Eigen::MatrixXi F(mesh_.facets.nb(), 3);

			for (int v = 0; v < V.rows(); v++)
			{
				const double *ptr = mesh_.vertices.point_ptr(v);
				V.row(v) << ptr[0], ptr[1];
			}

			for (int f = 0; f < F.rows(); f++)
				for (int i = 0; i < F.cols(); i++)
					F(f, i) = mesh_.facets.vertex(f, i);

			n_elements = 0;
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

		void NCMesh2D::bounding_box(RowVectorNd &min, RowVectorNd &max) const
		{
			min = vertices[0].pos;
			max = vertices[0].pos;

			for (const auto &v : vertices)
			{
				for (int d = 0; d < 2; d++)
				{
					if (v.pos[d] > max[d])
						max[d] = v.pos[d];
					if (v.pos[d] < min[d])
						min[d] = v.pos[d];
				}
			}
		}

		Navigation::Index NCMesh2D::get_index_from_face(int f, int lv) const
		{
			const auto &elem = elements[valid_to_all_elem(f)];

			Navigation::Index idx2;
			idx2.face = f;
			idx2.vertex = all_to_valid_vertex(elem.vertices(lv));
			idx2.edge = all_to_valid_edge(elem.edges(lv));
			idx2.face_corner = -1;

			return idx2;
		}

		Navigation::Index NCMesh2D::switch_vertex(Navigation::Index idx) const
		{
			const auto &elem = elements[valid_to_all_elem(idx.face)];
			const auto &edge = edges[valid_to_all_edge(idx.edge)];

			Navigation::Index idx2;
			idx2.face = idx.face;
			idx2.edge = idx.edge;

			int v1 = valid_to_all_vertex(idx.vertex);
			int v2 = -1;
			for (int i = 0; i < edge.vertices.size(); i++)
				if (edge.vertices(i) != v1)
				{
					v2 = edge.vertices(i);
					break;
				}

			idx2.vertex = all_to_valid_vertex(v2);
			idx2.face_corner = -1;

			return idx2;
		}

		Navigation::Index NCMesh2D::switch_edge(Navigation::Index idx) const
		{
			const auto &elem = elements[valid_to_all_elem(idx.face)];

			Navigation::Index idx2;
			idx2.face = idx.face;
			idx2.vertex = idx.vertex;
			idx2.face_corner = -1;

			for (int i = 0; i < elem.edges.size(); i++)
			{
				const auto &edge = edges[elem.edges(i)];
				const int valid_edge_id = all_to_valid_edge(elem.edges(i));
				if (valid_edge_id != idx.edge && find(edge.vertices, idx.vertex) >= 0)
				{
					idx2.edge = valid_edge_id;
					break;
				}
			}

			return idx2;
		}

		Navigation::Index NCMesh2D::switch_face(Navigation::Index idx) const
		{
			Navigation::Index idx2;
			idx2.edge = idx.edge;
			idx2.vertex = idx.vertex;
			idx2.face_corner = -1;

			const auto &edge = edges[valid_to_all_edge(idx.edge)];
			if (edge.n_elem() == 2)
				idx2.face = (edge.find_opposite_element(valid_to_all_elem(idx.face)));
			else
				idx2.face = -1;

			return idx2;
		}

		void NCMesh2D::normalize()
		{
			polyfem::RowVectorNd min, max;
			bounding_box(min, max);

			auto extent = max - min;
			double scale = std::max(extent(0), extent(1));

			for (auto &v : vertices)
				v.pos = (v.pos - min.transpose()) / scale;
		}

		double NCMesh2D::edge_length(const int gid) const
		{
			const int v1 = edge_vertex(gid, 0);
			const int v2 = edge_vertex(gid, 1);

			return (point(v1) - point(v2)).norm();
		}

		void NCMesh2D::compute_elements_tag()
		{
			elements_tag_.assign(n_faces(), ElementType::SIMPLEX);
		}
		void NCMesh2D::update_elements_tag()
		{
			elements_tag_.assign(n_faces(), ElementType::SIMPLEX);
		}

		void NCMesh2D::set_point(const int global_index, const RowVectorNd &p)
		{
			vertices[valid_to_all_vertex(global_index)].pos = p;
		}

		RowVectorNd NCMesh2D::edge_barycenter(const int index) const
		{
			const int v1 = edge_vertex(index, 0);
			const int v2 = edge_vertex(index, 1);

			return 0.5 * (point(v1) + point(v2));
		}

		// void NCMesh2D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const
		// {
		// 	ranges.clear();

		// 	std::vector<Eigen::MatrixXi> local_tris(n_faces());
		// 	std::vector<Eigen::MatrixXd> local_pts(n_faces());

		// 	int total_tris = 0;
		// 	int total_pts = 0;

		// 	ranges.push_back(0);

		// 	for (int f = 0; f < n_faces(); ++f)
		// 	{
		// 		const int n_vertices = n_face_vertices(f);

		// 		Eigen::MatrixXd face_pts(n_vertices, 2);
		// 		local_tris[f].resize(n_vertices - 2, 3);

		// 		for (int i = 0; i < n_vertices; ++i)
		// 		{
		// 			const int vertex = face_vertex(f, i);
		// 			auto pt = point(vertex);
		// 			face_pts(i, 0) = pt[0];
		// 			face_pts(i, 1) = pt[1];
		// 		}

		// 		for (int i = 1; i < n_vertices - 1; ++i)
		// 		{
		// 			local_tris[f].row(i - 1) << 0, i, i + 1;
		// 		}

		// 		local_pts[f] = face_pts;

		// 		total_tris += local_tris[f].rows();
		// 		total_pts += local_pts[f].rows();

		// 		ranges.push_back(total_tris);

		// 		assert(local_pts[f].rows() == face_pts.rows());
		// 	}

		// 	tris.resize(total_tris, 3);
		// 	pts.resize(total_pts, 2);

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

		void NCMesh2D::set_body_ids(const std::vector<int> &body_ids)
		{
			assert(body_ids.size() == n_faces());
			for (int i = 0; i < body_ids.size(); i++)
			{
				elements[valid_to_all_elem(i)].body_id = body_ids[i];
			}
		}

		void NCMesh2D::set_boundary_ids(const std::vector<int> &boundary_ids)
		{
			assert(boundary_ids.size() == n_edges());
			for (int i = 0; i < boundary_ids.size(); i++)
			{
				edges[valid_to_all_edge(i)].boundary_id = boundary_ids[i];
			}
		}

		void NCMesh2D::compute_body_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &)> &marker)
		{
			body_ids_.resize(n_faces());
			std::fill(body_ids_.begin(), body_ids_.end(), -1);

			for (int e = 0; e < n_faces(); ++e)
			{
				const auto bary = face_barycenter(e);
				body_ids_[e] = marker(e, element_vertices(e), bary);
				elements[valid_to_all_elem(e)].body_id = body_ids_[e];
			}
		}

		void NCMesh2D::compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker)
		{
			boundary_ids_.resize(n_edges());

			for (int e = 0; e < n_edges(); ++e)
			{
				bool is_boundary = is_boundary_edge(e);
				const auto p = edge_barycenter(e);

				std::vector<int> vs = {edge_vertex(e, 0), edge_vertex(e, 1)};
				std::sort(vs.begin(), vs.end());
				boundary_ids_[e] = marker(e, vs, p, is_boundary);
				edges[valid_to_all_edge(e)].boundary_id = boundary_ids_[e];
			}
		}

	} // namespace mesh
} // namespace polyfem