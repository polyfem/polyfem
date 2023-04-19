#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/barycentric_coordinates.h>

#include <geogram/mesh/mesh_io.h>
#include <fstream>

namespace polyfem
{
	using namespace utils;

	namespace mesh
	{
		double Mesh3D::tri_area(const int gid) const
		{
			const int n_vertices = n_face_vertices(gid);
			assert(n_vertices == 3);

			const auto v1 = point(face_vertex(gid, 0));
			const auto v2 = point(face_vertex(gid, 1));
			const auto v3 = point(face_vertex(gid, 2));

			const Eigen::Vector3d e0 = (v2 - v1).transpose();
			const Eigen::Vector3d e1 = (v3 - v1).transpose();

			return e0.cross(e1).norm() / 2;
		}

		void Mesh3D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const
		{
			p0.resize(n_edges(), 3);
			p1.resize(p0.rows(), p0.cols());

			for (std::size_t e = 0; e < n_edges(); ++e)
			{
				const int v0 = edge_vertex(e, 0);
				const int v1 = edge_vertex(e, 1);

				p0.row(e) = point(v0);
				p1.row(e) = point(v1);
			}
		}

		void Mesh3D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const
		{
			int count = 0;
			for (size_t i = 0; i < valid_elements.size(); ++i)
			{
				if (valid_elements[i])
				{
					count += n_cell_edges(i);
				}
			}

			p0.resize(count, 3);
			p1.resize(count, 3);

			count = 0;

			for (size_t i = 0; i < valid_elements.size(); ++i)
			{
				if (!valid_elements[i])
					continue;

				for (size_t ei = 0; ei < n_cell_edges(i); ++ei)
				{
					const int e = cell_edge(i, ei);
					p0.row(count) = point(edge_vertex(e, 0));
					p1.row(count) = point(edge_vertex(e, 1));

					++count;
				}
			}
		}

		RowVectorNd Mesh3D::edge_node(const Navigation3D::Index &index, const int n_new_nodes, const int i) const
		{
			if (orders_.size() <= 0 || orders_(index.element) == 1 || edge_nodes_.empty() || edge_nodes_[index.edge].nodes.rows() != n_new_nodes)
			{
				const auto v1 = point(index.vertex);
				const auto v2 = point(switch_vertex(index).vertex);

				const double t = i / (n_new_nodes + 1.0);

				return (1 - t) * v1 + t * v2;
			}

			const auto &n = edge_nodes_[index.edge];
			if (n.v1 == index.vertex)
				return n.nodes.row(i - 1);
			else
				return n.nodes.row(n.nodes.rows() - i);
		}

		RowVectorNd Mesh3D::face_node(const Navigation3D::Index &index, const int n_new_nodes, const int i, const int j) const
		{
			const int tmp = n_new_nodes == 2 ? 3 : n_new_nodes;
			if (is_simplex(index.element))
			{
				if (orders_.size() <= 0 || orders_(index.element) == 1 || orders_(index.element) == 2 || face_nodes_.empty() || face_nodes_[index.face].nodes.rows() != tmp)
				{
					const auto v1 = point(index.vertex);
					const auto v2 = point(switch_vertex(index).vertex);
					const auto v3 = point(switch_vertex(switch_edge(index)).vertex);

					const double b2 = i / (n_new_nodes + 2.0);
					const double b3 = j / (n_new_nodes + 2.0);
					const double b1 = 1 - b3 - b2;
					assert(b3 < 1);
					assert(b3 > 0);

					return b1 * v1 + b2 * v2 + b3 * v3;
				}

				const int ii = i - 1;
				const int jj = j - 1;

				assert(orders_(index.element) == 3 || orders_(index.element) == 4);
				const auto &n = face_nodes_[index.face];
				int lindex = jj * n_new_nodes + ii;

				if (orders_(index.element) == 4)
				{
					static const std::array<int, 3> remapping = {{0, 2, 1}};
					if (n.v1 == index.vertex)
					{
						if (n.v2 != next_around_face(index).vertex)
						{
							lindex = remapping[lindex];
							assert(n.v3 == next_around_face(index).vertex);
						}
						else
						{
							assert(n.v2 == next_around_face(index).vertex);
						}
					}
					else if (n.v2 == index.vertex)
					{

						if (n.v3 != next_around_face(index).vertex)
						{
							lindex = remapping[lindex];
							assert(n.v1 == next_around_face(index).vertex);
						}
						else
						{
							assert(n.v3 == switch_vertex(index).vertex);
						}

						lindex = (lindex + 1) % 3;
					}
					else if (n.v3 == index.vertex)
					{

						if (n.v1 != next_around_face(index).vertex)
						{
							lindex = remapping[lindex];
							assert(n.v2 == next_around_face(index).vertex);
						}
						else
						{
							assert(n.v1 == switch_vertex(index).vertex);
						}

						lindex = (lindex + 2) % 3;
					}
					else
					{
						// assert(false);
					}
				}

				return n.nodes.row(lindex);
			}
			else if (is_cube(index.element))
			{
				// supports only blilinear quads
				assert(orders_.size() <= 0 || orders_(index.element) == 1);

				const auto v1 = point(index.vertex);
				const auto v2 = point(switch_vertex(index).vertex);
				const auto v3 = point(switch_vertex(switch_edge(switch_vertex(index))).vertex);
				const auto v4 = point(switch_vertex(switch_edge(index)).vertex);

				const double b1 = i / (n_new_nodes + 1.0);
				const double b2 = j / (n_new_nodes + 1.0);

				return v1 * (1 - b1) * (1 - b2) + v2 * b1 * (1 - b2) + v3 * b1 * b2 + v4 * (1 - b1) * b2;
			}

			assert(false);
			return RowVectorNd(3, 1);
		}

		RowVectorNd Mesh3D::cell_node(const Navigation3D::Index &index, const int n_new_nodes, const int i, const int j, const int k) const
		{
			if (is_simplex(index.element) && orders_.size() > 0 && orders_(index.element) == n_new_nodes + 3)
			{
				assert(n_new_nodes == 1); // test higher than 4 order meshes
				const auto &n = cell_nodes_[index.element];
				assert(n.nodes.rows() == 1);
				return n.nodes;
			}

			if (n_new_nodes == 1)
				return cell_barycenter(index.element);

			if (is_simplex(index.element))
			{
				if (n_new_nodes == 1)
					return cell_barycenter(index.element);
				else
				{
					const auto v1 = point(index.vertex);
					const auto v2 = point(switch_vertex(index).vertex);
					const auto v3 = point(switch_vertex(switch_edge(switch_vertex(index))).vertex);
					const auto v4 = point(switch_vertex(switch_edge(switch_face(index))).vertex);

					const double w1 = double(i) / (n_new_nodes + 3);
					const double w2 = double(j) / (n_new_nodes + 3);
					const double w3 = double(k) / (n_new_nodes + 3);
					const double w4 = 1 - w1 - w2 - w3;

					// return v1 * w3
					// 	   + v2 * w4
					// 	   + v3 * w1
					// 	   + v4 * w2;

					return w4 * v1 + w1 * v2 + w2 * v3 + w3 * v4;
				}
			}
			else if (is_cube(index.element))
			{
				// supports only blilinear hexes
				assert(orders_.size() <= 0 || orders_(index.element) == 1);

				const auto v1 = point(index.vertex);
				const auto v2 = point(switch_vertex(index).vertex);
				const auto v3 = point(switch_vertex(switch_edge(switch_vertex(index))).vertex);
				const auto v4 = point(switch_vertex(switch_edge(index)).vertex);

				const Navigation3D::Index index1 = switch_face(switch_edge(switch_vertex(switch_edge(switch_face(index)))));
				const auto v5 = point(index1.vertex);
				const auto v6 = point(switch_vertex(index1).vertex);
				const auto v7 = point(switch_vertex(switch_edge(switch_vertex(index1))).vertex);
				const auto v8 = point(switch_vertex(switch_edge(index1)).vertex);

				const double b1 = i / (n_new_nodes + 1.0);
				const double b2 = j / (n_new_nodes + 1.0);

				const double b3 = k / (n_new_nodes + 1.0);

				RowVectorNd blin1 = v1 * (1 - b1) * (1 - b2) + v2 * b1 * (1 - b2) + v3 * b1 * b2 + v4 * (1 - b1) * b2;
				RowVectorNd blin2 = v5 * (1 - b1) * (1 - b2) + v6 * b1 * (1 - b2) + v7 * b1 * b2 + v8 * (1 - b1) * b2;

				return (1 - b3) * blin1 + b3 * blin2;
			}

			assert(false);
			return RowVectorNd(3, 1);
		}

		void Mesh3D::to_face_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> &to_face) const
		{
			// top
			to_face[0] = [&](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(switch_edge(switch_face(idx))))); };
			// bottom
			to_face[1] = [&](Navigation3D::Index idx) { return idx; };

			// left
			to_face[2] = [&](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(idx))); };
			// right
			to_face[3] = [&](Navigation3D::Index idx) { return switch_face(switch_edge(idx)); };

			// back
			to_face[4] = [&](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx))))); };
			// front
			to_face[5] = [&](Navigation3D::Index idx) { return switch_face(idx); };
		}

		void Mesh3D::to_vertex_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> &to_vertex) const
		{
			to_vertex[0] = [&](Navigation3D::Index idx) { return idx; };
			to_vertex[1] = [&](Navigation3D::Index idx) { return switch_vertex(idx); };
			to_vertex[2] = [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_vertex(idx))); };
			to_vertex[3] = [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(idx)); };

			to_vertex[4] = [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(idx))); };
			to_vertex[5] = [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_vertex(switch_edge(switch_face(idx))))); };
			to_vertex[6] = [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(switch_vertex(switch_edge(switch_vertex(idx)))))); };
			to_vertex[7] = [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(switch_vertex(switch_edge(idx))))); };
		}

		void Mesh3D::to_edge_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> &to_edge) const
		{
			to_edge[0] = [&](Navigation3D::Index idx) { return idx; };
			to_edge[1] = [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(idx)); };
			to_edge[2] = [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))); };
			to_edge[3] = [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))); };

			to_edge[4] = [&](Navigation3D::Index idx) { return switch_edge(switch_face(idx)); };
			to_edge[5] = [&](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(idx)))); };
			to_edge[6] = [&](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))); };
			to_edge[7] = [&](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))); };

			to_edge[8] = [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(idx)))); };
			to_edge[9] = [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(idx)))))); };
			to_edge[10] = [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))); };
			to_edge[11] = [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))))); };
		}

		//   v7────v6
		//   ╱┆    ╱│
		// v4─┼──v5 │
		//  │v3┄┄┄┼v2
		//  │╱    │╱
		// v0────v1
		std::array<int, 8> Mesh3D::get_ordered_vertices_from_hex(const int element_index) const
		{
			assert(is_cube(element_index));
			auto idx = get_index_from_element(element_index);
			std::array<int, 8> v;

			std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> to_vertex;
			to_vertex_functions(to_vertex);
			for (int i = 0; i < 8; ++i)
				v[i] = to_vertex[i](idx).vertex;

			// for (int lv = 0; lv < 4; ++lv) {
			// 	v[lv] = idx.vertex;
			// 	idx = next_around_face_of_element(idx);
			// }
			// // assert(idx == get_index_from_element(element_index));
			// idx = switch_face(switch_edge(switch_vertex(switch_edge(switch_face(idx)))));
			// for (int lv = 0; lv < 4; ++lv) {
			// 	v[4+lv] = idx.vertex;
			// 	idx = next_around_face_of_element(idx);
			// }
			return v;
		}

		std::array<int, 4> Mesh3D::get_ordered_vertices_from_tet(const int element_index) const
		{
			auto idx = get_index_from_element(element_index);
			std::array<int, 4> v;

			for (int lv = 0; lv < 3; ++lv)
			{
				v[lv] = idx.vertex;
				idx = next_around_face(idx);
			}
			// assert(idx == get_index_from_element(element_index));
			idx = switch_vertex(switch_edge(switch_face(idx)));
			v[3] = idx.vertex;

			// std::array<GEO::vec3, 4> vertices;

			// for(int lv = 0; lv < 4; ++lv)
			// {
			// 	auto pt = point(v[lv]);
			// 	for(int d = 0; d < 3; ++d)
			// 	{
			// 		vertices[lv][d] = pt(d);
			// 	}
			// }

			// const double vol = GEO::Geom::tetra_signed_volume(vertices[0], vertices[1], vertices[2], vertices[3]);
			// if(vol < 0)
			// {
			// 	std::cout << "negative vol" << std::endl;
			// //	idx = switch_vertex(get_index_from_element(element_index));
			// //	for (int lv = 0; lv < 3; ++lv) {
			// //		v[lv] = idx.vertex;
			// //		idx = next_around_face(idx);
			// //	}
			// //// assert(idx == get_index_from_element(element_index));
			// //	idx = switch_vertex(switch_edge(switch_face(idx)));
			// //	v[3] = idx.vertex;
			// }

			return v;
		}

		void Mesh3D::elements_boxes(std::vector<std::array<Eigen::Vector3d, 2>> &boxes) const
		{
			boxes.resize(n_elements());

			for (int i = 0; i < n_elements(); ++i)
			{
				auto &box = boxes[i];
				box[0].setConstant(std::numeric_limits<double>::max());
				box[1].setConstant(std::numeric_limits<double>::min());

				for (int j = 0; j < n_cell_vertices(i); ++j)
				{
					const int v_id = cell_vertex(i, j);
					for (int d = 0; d < 3; ++d)
					{
						box[0][d] = std::min(box[0][d], point(v_id)[d]);
						box[1][d] = std::max(box[1][d], point(v_id)[d]);
					}
				}
			}
		}

		void Mesh3D::barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coord) const
		{
			assert(is_simplex(el_id));

			const auto indices = get_ordered_vertices_from_tet(el_id);

			const auto A = point(indices[0]);
			const auto B = point(indices[1]);
			const auto C = point(indices[2]);
			const auto D = point(indices[3]);

			igl::barycentric_coordinates(p, A, B, C, D, coord);
		}

		void Mesh3D::compute_cell_jacobian(const int el_id, const Eigen::MatrixXd &reference_map, Eigen::MatrixXd &jacobian) const
		{
			assert(is_simplex(el_id));

			const auto indices = get_ordered_vertices_from_tet(el_id);

			const auto A = point(indices[0]);
			const auto B = point(indices[1]);
			const auto C = point(indices[2]);
			const auto D = point(indices[3]);

			Eigen::MatrixXd coords(4, 4);
			coords << A, 1, B, 1, C, 1, D, 1;
			coords.transposeInPlace();

			jacobian = coords * reference_map;

			assert(jacobian.determinant() > 0);
		}

	} // namespace mesh
} // namespace polyfem
