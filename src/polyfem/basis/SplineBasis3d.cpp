#include "SplineBasis3d.hpp"

#include "LagrangeBasis3d.hpp"
#include "function/QuadraticBSpline3d.hpp"
#include <polyfem/quadrature/HexQuadrature.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polysolve/linear/Solver.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/utils/Types.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <Eigen/Sparse>

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <numeric>

// TODO carefull with simplices

namespace polyfem
{
	using namespace Eigen;
	using namespace polysolve;
	using namespace assembler;
	using namespace mesh;
	using namespace quadrature;

	namespace basis
	{

		namespace
		{
			class SpaceMatrix
			{
			public:
				inline const int &operator()(const int i, const int j, const int k) const
				{
					return space_[k](i, j);
				}

				inline int &operator()(const int i, const int j, const int k)
				{
					return space_[k](i, j);
				}

				bool is_k_regular = false;
				int x, y, z;
				int edge_id;

				bool is_regular(const int xx, const int yy, const int zz) const
				{
					if (!is_k_regular)
						return true;

					return !((x == 1 && y == yy && z == zz) || (x == xx && y == 1 && z == zz) || (x == xx && y == yy && z == 1));
					// space(1, y, z).size() <= 1 && space(x, 1, z).size() <= 1 && space(x, y, 1).size() <= 1
				}

			private:
				std::array<Matrix<int, 3, 3>, 3> space_;
			};

			bool is_edge_singular(const Navigation3D::Index &index, const Mesh3D &mesh)
			{
				std::vector<int> ids;
				mesh.get_edge_elements_neighs(index.edge, ids);

				if (ids.size() == 4 || mesh.is_boundary_edge(index.edge))
					return false;

				for (auto idx : ids)
				{
					if (mesh.is_polytope(idx))
						return false;
				}

				return true;
			}

			void print_local_space(const SpaceMatrix &space)
			{
				std::stringstream ss;
				for (int k = 2; k >= 0; --k)
				{
					for (int j = 2; j >= 0; --j)
					{
						for (int i = 0; i < 3; ++i)
						{
							// if(space(i, j, k).size() > 0){
							// for(std::size_t l = 0; l < space(i, j, k).size(); ++l)
							ss << space(i, j, k) << "\t";
							// }
							// else
							// ss<<"x\t";
						}
						ss << std::endl;
					}

					ss << "\n"
					   << std::endl;
				}

				logger().trace("Local space:\n{}", ss.str());
			}

			int node_id_from_face_index(const Mesh3D &mesh, MeshNodes &mesh_nodes, const Navigation3D::Index &index)
			{
				int el_id = mesh.switch_element(index).element;
				if (el_id >= 0 && mesh.is_cube(el_id))
				{
					return mesh_nodes.node_id_from_cell(el_id);
				}

				return mesh_nodes.node_id_from_face(index.face);
			}

			int node_id_from_edge_index(const Mesh3D &mesh, MeshNodes &mesh_nodes, const Navigation3D::Index &index)
			{
				Navigation3D::Index new_index = mesh.switch_element(index);
				int el_id = new_index.element;

				if (el_id < 0 || mesh.is_polytope(el_id))
				{
					new_index = mesh.switch_element(mesh.switch_face(index));
					el_id = new_index.element;

					if (el_id < 0 || mesh.is_polytope(el_id))
					{
						return mesh_nodes.node_id_from_edge(index.edge);
					}

					return node_id_from_face_index(mesh, mesh_nodes, mesh.switch_face(new_index));
				}

				return node_id_from_face_index(mesh, mesh_nodes, mesh.switch_face(new_index));
			}

			int node_id_from_vertex_index_explore(const Mesh3D &mesh, const MeshNodes &mesh_nodes, const Navigation3D::Index &index, int &node_id)
			{
				Navigation3D::Index new_index = mesh.switch_element(index);

				int id = new_index.element;

				if (id < 0 || mesh.is_polytope(id))
				{
					// id = vertex_node_id(index.vertex);
					// node = node_from_vertex(index.vertex);
					node_id = mesh_nodes.primitive_from_vertex(index.vertex);
					return 3;
				}

				new_index = mesh.switch_element(mesh.switch_face(new_index));
				id = new_index.element;

				if (id < 0 || mesh.is_polytope(id))
				{
					// id = edge_node_id(switch_edge(new_index).edge);
					// node = node_from_edge(switch_edge(new_index).edge);
					node_id = mesh_nodes.primitive_from_edge(mesh.switch_edge(new_index).edge);
					return 2;
				}

				new_index = mesh.switch_element(mesh.switch_face(mesh.switch_edge(new_index)));
				id = new_index.element;

				if (id < 0 || mesh.is_polytope(id))
				{
					// id = face_node_id(new_index.face);
					// node = node_from_face(new_index.face);
					node_id = mesh_nodes.primitive_from_face(new_index.face);
					return 1;
				}

				// node = node_from_element(id);
				node_id = mesh_nodes.primitive_from_cell(id);
				return 0;
			}

			int node_id_from_vertex_index(const Mesh3D &mesh, MeshNodes &mesh_nodes, const Navigation3D::Index &index)
			{
				std::array<int, 6> path;
				std::array<int, 6> primitive_ids;

				path[0] = node_id_from_vertex_index_explore(mesh, mesh_nodes, index, primitive_ids[0]);
				path[1] = node_id_from_vertex_index_explore(mesh, mesh_nodes, mesh.switch_face(index), primitive_ids[1]);

				path[2] = node_id_from_vertex_index_explore(mesh, mesh_nodes, mesh.switch_edge(index), primitive_ids[2]);
				path[3] = node_id_from_vertex_index_explore(mesh, mesh_nodes, mesh.switch_face(mesh.switch_edge(index)), primitive_ids[3]);

				path[4] = node_id_from_vertex_index_explore(mesh, mesh_nodes, mesh.switch_edge(mesh.switch_face(index)), primitive_ids[4]);
				path[5] = node_id_from_vertex_index_explore(mesh, mesh_nodes, mesh.switch_face(mesh.switch_edge(mesh.switch_face(index))), primitive_ids[5]);

				const int min_path = *std::min_element(path.begin(), path.end());

				int primitive_id = 0;
				for (int i = 0; i < 6; ++i)
				{
					if (path[i] == min_path)
					{
						primitive_id = primitive_ids[i];
						break;
					}
				}

				return mesh_nodes.node_id_from_primitive(primitive_id);
			}

			void get_edge_elements_neighs(const Mesh3D &mesh, MeshNodes &mesh_nodes, const int element_id, const int edge_id, int dir, std::vector<int> &ids)
			{
				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> to_edge;
				mesh.to_edge_functions(to_edge);

				Navigation3D::Index index;
				for (int i = 0; i < 12; ++i)
				{
					index = to_edge[i](mesh.get_index_from_element(element_id));

					if (index.edge == edge_id)
						break;
				}

				assert(index.edge == edge_id);

				if (dir == 1)
				{
					int id;
					do
					{
						ids.push_back(mesh_nodes.node_id_from_cell(index.element));
						index = mesh.next_around_edge(index);
					} while (index.element != element_id);

					return;
				}

				if (dir == 0)
				{
					int id;
					do
					{
						const Navigation3D::Index f_index = mesh.switch_face(mesh.switch_edge(index));
						ids.push_back(node_id_from_face_index(mesh, mesh_nodes, f_index));

						index = mesh.next_around_edge(index);
					} while (index.element != element_id);

					return;
				}

				if (dir == 2)
				{
					int id;
					do
					{
						const Navigation3D::Index f_index = mesh.switch_face(mesh.switch_edge(mesh.switch_vertex(index)));
						ids.push_back(node_id_from_face_index(mesh, mesh_nodes, f_index));

						index = mesh.next_around_edge(index);
					} while (index.element != element_id);

					return;
				}

				assert(false);
			}

			void add_edge_id_for_poly(const Navigation3D::Index &index, const Mesh3D &mesh, MeshNodes &mesh_nodes, const int global_index, std::map<int, InterfaceData> &poly_face_to_data)
			{
				const int f1 = index.face;
				const int f2 = mesh.switch_face(index).face;

				const int id1 = mesh_nodes.primitive_from_face(f1);
				const int id2 = mesh_nodes.primitive_from_face(f2);

				if (mesh_nodes.is_primitive_interface(id1))
				{
					InterfaceData &data = poly_face_to_data[f1];
					data.local_indices.push_back(global_index);
				}

				if (mesh_nodes.is_primitive_interface(id2))
				{
					InterfaceData &data = poly_face_to_data[f2];
					data.local_indices.push_back(global_index);
				}
			}

			void add_vertex_id_for_poly(const Navigation3D::Index &index, const Mesh3D &mesh, MeshNodes &mesh_nodes, const int global_index, std::map<int, InterfaceData> &poly_face_to_data)
			{
				const int f1 = index.face;
				const int f2 = mesh.switch_face(index).face;
				const int f3 = mesh.switch_face(mesh.switch_edge(index)).face;

				const int id1 = mesh_nodes.primitive_from_face(f1);
				const int id2 = mesh_nodes.primitive_from_face(f2);
				const int id3 = mesh_nodes.primitive_from_face(f3);

				if (mesh_nodes.is_primitive_interface(id1))
				{
					InterfaceData &data = poly_face_to_data[f1];
					data.local_indices.push_back(global_index);
				}

				if (mesh_nodes.is_primitive_interface(id2))
				{
					InterfaceData &data = poly_face_to_data[f2];
					data.local_indices.push_back(global_index);
				}

				if (mesh_nodes.is_primitive_interface(id3))
				{
					InterfaceData &data = poly_face_to_data[f3];
					data.local_indices.push_back(global_index);
				}
			}

			void explore_edge(const Navigation3D::Index &index, const Mesh3D &mesh, MeshNodes &mesh_nodes, const int x, const int y, const int z, SpaceMatrix &space, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_face_to_data)
			{
				int node_id = node_id_from_edge_index(mesh, mesh_nodes, index);
				space(x, y, z) = node_id;
				// node(x, y, z) = mesh_nodes.node_position(node_id);

				if (is_edge_singular(index, mesh))
				{
					std::vector<int> ids;
					mesh.get_edge_elements_neighs(index.edge, ids);
					// irregular edge

					assert(!space.is_k_regular);
					space.is_k_regular = true;
					space.x = x;
					space.y = y;
					space.z = z;
					space.edge_id = index.edge;
				}

				// if(mesh_nodes.is_boundary(node_id))
				// bounday_nodes.push_back(node_id);

				add_edge_id_for_poly(index, mesh, mesh_nodes, 9 * z + 3 * y + x, poly_face_to_data);
			}

			void explore_vertex(const Navigation3D::Index &index, const Mesh3D &mesh, MeshNodes &mesh_nodes, const int x, const int y, const int z, SpaceMatrix &space, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_face_to_data)
			{
				int node_id = node_id_from_vertex_index(mesh, mesh_nodes, index);
				space(x, y, z) = node_id;
				// node(x, y, z) = mesh_nodes.node_position(node_id);

				// if(mesh_nodes.is_boundary(node_id))
				// bounday_nodes.push_back(node_id);

				add_vertex_id_for_poly(index, mesh, mesh_nodes, 9 * z + 3 * y + x, poly_face_to_data);
			}

			void explore_face(const Navigation3D::Index &index, const Mesh3D &mesh, MeshNodes &mesh_nodes, const int x, const int y, const int z, SpaceMatrix &space, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_face_to_data)
			{
				int node_id = node_id_from_face_index(mesh, mesh_nodes, index);
				space(x, y, z) = node_id;
				// node(x, y, z) = mesh_nodes.node_position(node_id);

				if (mesh_nodes.is_boundary(node_id))
				{
					// local_boundary.add_boundary_primitive(index.face, LagrangeBasis3d::quadr_hex_face_local_nodes(mesh, index)[8]-20);
					local_boundary.add_boundary_primitive(index.face, LagrangeBasis3d::hex_face_local_nodes(false, 2, mesh, index)[8] - 20);
					// bounday_nodes.push_back(node_id);
				}
				else if (mesh_nodes.is_interface(node_id))
				{
					InterfaceData &data = poly_face_to_data[index.face];
					data.local_indices.push_back(9 * z + 3 * y + x);
					// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
					// viewer.data.add_points(mesh_nodes.node_position(node_id), Eigen::MatrixXd::Constant(1, 3, 0));
				}
			}

			void build_local_space(const Mesh3D &mesh, MeshNodes &mesh_nodes, const int el_index, SpaceMatrix &space, std::vector<LocalBoundary> &local_boundary, std::map<int, InterfaceData> &poly_face_to_data)
			{
				assert(mesh.is_volume());

				Navigation3D::Index start_index = mesh.get_index_from_element(el_index);
				Navigation3D::Index index;

				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
				mesh.to_face_functions(to_face);

				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> to_edge;
				mesh.to_edge_functions(to_edge);

				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> to_vertex;
				mesh.to_vertex_functions(to_vertex);

				const int node_id = mesh_nodes.node_id_from_cell(el_index);
				space(1, 1, 1) = node_id;
				// node(1, 1, 1) = mesh_nodes.node_position(node_id);

				LocalBoundary lb(el_index, BoundaryType::QUAD);

				///////////////////////
				index = to_face[1](start_index);
				explore_face(index, mesh, mesh_nodes, 1, 1, 0, space, lb, poly_face_to_data);

				index = to_face[0](start_index);
				explore_face(index, mesh, mesh_nodes, 1, 1, 2, space, lb, poly_face_to_data);

				index = to_face[3](start_index);
				explore_face(index, mesh, mesh_nodes, 0, 1, 1, space, lb, poly_face_to_data);

				index = to_face[2](start_index);
				explore_face(index, mesh, mesh_nodes, 2, 1, 1, space, lb, poly_face_to_data);

				index = to_face[5](start_index);
				explore_face(index, mesh, mesh_nodes, 1, 0, 1, space, lb, poly_face_to_data);

				index = to_face[4](start_index);
				explore_face(index, mesh, mesh_nodes, 1, 2, 1, space, lb, poly_face_to_data);

				///////////////////////
				index = to_edge[0](start_index);
				explore_edge(index, mesh, mesh_nodes, 1, 0, 0, space, lb, poly_face_to_data);

				index = to_edge[1](start_index);
				explore_edge(index, mesh, mesh_nodes, 2, 1, 0, space, lb, poly_face_to_data);

				index = to_edge[2](start_index);
				explore_edge(index, mesh, mesh_nodes, 1, 2, 0, space, lb, poly_face_to_data);

				index = to_edge[3](start_index);
				explore_edge(index, mesh, mesh_nodes, 0, 1, 0, space, lb, poly_face_to_data);

				index = to_edge[4](start_index);
				explore_edge(index, mesh, mesh_nodes, 0, 0, 1, space, lb, poly_face_to_data);

				index = to_edge[5](start_index);
				explore_edge(index, mesh, mesh_nodes, 2, 0, 1, space, lb, poly_face_to_data);

				index = to_edge[6](start_index);
				explore_edge(index, mesh, mesh_nodes, 2, 2, 1, space, lb, poly_face_to_data);

				index = to_edge[7](start_index);
				explore_edge(index, mesh, mesh_nodes, 0, 2, 1, space, lb, poly_face_to_data);

				index = to_edge[8](start_index);
				explore_edge(index, mesh, mesh_nodes, 1, 0, 2, space, lb, poly_face_to_data);

				index = to_edge[9](start_index);
				explore_edge(index, mesh, mesh_nodes, 2, 1, 2, space, lb, poly_face_to_data);

				index = to_edge[10](start_index);
				explore_edge(index, mesh, mesh_nodes, 1, 2, 2, space, lb, poly_face_to_data);

				index = to_edge[11](start_index);
				explore_edge(index, mesh, mesh_nodes, 0, 1, 2, space, lb, poly_face_to_data);

				////////////////////////////////////////////////////////////////////////
				index = to_vertex[0](start_index);
				explore_vertex(index, mesh, mesh_nodes, 0, 0, 0, space, lb, poly_face_to_data);

				index = to_vertex[1](start_index);
				explore_vertex(index, mesh, mesh_nodes, 2, 0, 0, space, lb, poly_face_to_data);

				index = to_vertex[2](start_index);
				explore_vertex(index, mesh, mesh_nodes, 2, 2, 0, space, lb, poly_face_to_data);

				index = to_vertex[3](start_index);
				explore_vertex(index, mesh, mesh_nodes, 0, 2, 0, space, lb, poly_face_to_data);

				index = to_vertex[4](start_index);
				explore_vertex(index, mesh, mesh_nodes, 0, 0, 2, space, lb, poly_face_to_data);

				index = to_vertex[5](start_index);
				explore_vertex(index, mesh, mesh_nodes, 2, 0, 2, space, lb, poly_face_to_data);

				index = to_vertex[6](start_index);
				explore_vertex(index, mesh, mesh_nodes, 2, 2, 2, space, lb, poly_face_to_data);

				index = to_vertex[7](start_index);
				explore_vertex(index, mesh, mesh_nodes, 0, 2, 2, space, lb, poly_face_to_data);

				if (!lb.empty())
					local_boundary.emplace_back(lb);
			}

			void setup_knots_vectors(MeshNodes &mesh_nodes, const SpaceMatrix &space, std::array<std::array<double, 4>, 3> &h_knots, std::array<std::array<double, 4>, 3> &v_knots, std::array<std::array<double, 4>, 3> &w_knots)
			{
				// left and right neigh are absent
				if (mesh_nodes.is_boundary_or_interface(space(0, 1, 1)) && mesh_nodes.is_boundary_or_interface(space(2, 1, 1)))
				{
					h_knots[0] = {{0, 0, 0, 1}};
					h_knots[1] = {{0, 0, 1, 1}};
					h_knots[2] = {{0, 1, 1, 1}};
				}
				// left neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(0, 1, 1)))
				{
					h_knots[0] = {{0, 0, 0, 1}};
					h_knots[1] = {{0, 0, 1, 2}};
					h_knots[2] = {{0, 1, 2, 3}};
				}
				// right neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(2, 1, 1)))
				{
					h_knots[0] = {{-2, -1, 0, 1}};
					h_knots[1] = {{-1, 0, 1, 1}};
					h_knots[2] = {{0, 1, 1, 1}};
				}
				else
				{
					h_knots[0] = {{-2, -1, 0, 1}};
					h_knots[1] = {{-1, 0, 1, 2}};
					h_knots[2] = {{0, 1, 2, 3}};
				}

				// top and bottom neigh are absent
				if (mesh_nodes.is_boundary_or_interface(space(1, 0, 1)) && mesh_nodes.is_boundary_or_interface(space(1, 2, 1)))
				{
					v_knots[0] = {{0, 0, 0, 1}};
					v_knots[1] = {{0, 0, 1, 1}};
					v_knots[2] = {{0, 1, 1, 1}};
				}
				// bottom neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(1, 0, 1)))
				{
					v_knots[0] = {{0, 0, 0, 1}};
					v_knots[1] = {{0, 0, 1, 2}};
					v_knots[2] = {{0, 1, 2, 3}};
				}
				// top neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(1, 2, 1)))
				{
					v_knots[0] = {{-2, -1, 0, 1}};
					v_knots[1] = {{-1, 0, 1, 1}};
					v_knots[2] = {{0, 1, 1, 1}};
				}
				else
				{
					v_knots[0] = {{-2, -1, 0, 1}};
					v_knots[1] = {{-1, 0, 1, 2}};
					v_knots[2] = {{0, 1, 2, 3}};
				}

				// front and back neigh are absent
				if (mesh_nodes.is_boundary_or_interface(space(1, 1, 0)) && mesh_nodes.is_boundary_or_interface(space(1, 1, 2)))
				{
					w_knots[0] = {{0, 0, 0, 1}};
					w_knots[1] = {{0, 0, 1, 1}};
					w_knots[2] = {{0, 1, 1, 1}};
				}
				// back neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(1, 1, 0)))
				{
					w_knots[0] = {{0, 0, 0, 1}};
					w_knots[1] = {{0, 0, 1, 2}};
					w_knots[2] = {{0, 1, 2, 3}};
				}
				// front neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(1, 1, 2)))
				{
					w_knots[0] = {{-2, -1, 0, 1}};
					w_knots[1] = {{-1, 0, 1, 1}};
					w_knots[2] = {{0, 1, 1, 1}};
				}
				else
				{
					w_knots[0] = {{-2, -1, 0, 1}};
					w_knots[1] = {{-1, 0, 1, 2}};
					w_knots[2] = {{0, 1, 2, 3}};
				}
			}

			void basis_for_regular_hex(MeshNodes &mesh_nodes, const SpaceMatrix &space, const std::array<std::array<double, 4>, 3> &h_knots, const std::array<std::array<double, 4>, 3> &v_knots, const std::array<std::array<double, 4>, 3> &w_knots, ElementBases &b)
			{
				for (int z = 0; z < 3; ++z)
				{
					for (int y = 0; y < 3; ++y)
					{
						for (int x = 0; x < 3; ++x)
						{
							if (space.is_regular(x, y, z)) // space(1, y, z).size() <= 1 && space(x, 1, z).size() <= 1 && space(x, y, 1).size() <= 1)
							{
								const int local_index = 9 * z + 3 * y + x;
								const int global_index = space(x, y, z);
								const auto node = mesh_nodes.node_position(global_index);
								// loc_nodes(x, y, z);

								b.bases[local_index].init(2, global_index, local_index, node);

								const QuadraticBSpline3d spline(h_knots[x], v_knots[y], w_knots[z]);

								b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
								b.bases[local_index].set_grad([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
							}
						}
					}
				}
			}

			void basis_for_irregulard_hex(const int el_index, const Mesh3D &mesh, MeshNodes &mesh_nodes, const SpaceMatrix &space, const std::array<std::array<double, 4>, 3> &h_knots, const std::array<std::array<double, 4>, 3> &v_knots, const std::array<std::array<double, 4>, 3> &w_knots, ElementBases &b, std::map<int, InterfaceData> &poly_face_to_data)
			{
				for (int z = 0; z < 3; ++z)
				{
					for (int y = 0; y < 3; ++y)
					{
						for (int x = 0; x < 3; ++x)
						{
							if (!space.is_regular(x, y, z)) // space(1, y, z).size() > 1 || space(x, 1, z).size() > 1 || space(x, y, 1).size() > 1)
							{
								const int local_index = 9 * z + 3 * y + x;

								int mpx = -1;
								int mpy = -1;
								int mpz = -1;

								int mmx = -1;
								int mmy = -1;
								int mmz = -1;

								int xx = 1;
								int yy = 1;
								int zz = 1;

								const int edge_id = space.edge_id;
								int dir = -1;

								if (space.x == x && space.y == y && space.z == 1)
								{
									mpx = 1;
									mpy = y;
									mpz = z;

									mmx = x;
									mmy = 1;
									mmz = z;

									zz = z;
									dir = z;
								}
								else if (space.x == x && space.y == 1 && space.z == z)
								{
									mpx = 1;
									mpy = y;
									mpz = z;

									mmx = x;
									mmy = y;
									mmz = 1;

									yy = y;
									dir = y;
								}
								else if (space.x == 1 && space.y == y && space.z == z)
								{
									mpx = x;
									mpy = y;
									mpz = 1;

									mmx = x;
									mmy = 1;
									mmz = z;

									xx = x;
									dir = x;
								}
								else
									assert(false);

								const auto &center = b.bases[zz * 9 + yy * 3 + xx].global().front();

								const auto &el1 = b.bases[mpz * 9 + mpy * 3 + mpx].global().front();
								const auto &el2 = b.bases[mmz * 9 + mmy * 3 + mmx].global().front();

								std::vector<int> ids;
								get_edge_elements_neighs(mesh, mesh_nodes, el_index, edge_id, dir, ids);

								if (ids.front() != center.index)
								{
									assert(dir != 1);
									ids.clear();
									get_edge_elements_neighs(mesh, mesh_nodes, el_index, edge_id, dir == 2 ? 0 : 2, ids);
								}

								assert(ids.front() == center.index);

								std::vector<int> other_indices;
								std::vector<Eigen::MatrixXd> other_nodes;
								for (size_t i = 0; i < ids.size(); ++i)
								{
									const int node_id = ids[i];
									if (node_id != center.index && node_id != el1.index && node_id != el2.index)
									{
										other_indices.push_back(node_id);
									}
								}

								auto &base = b.bases[local_index];

								const int k = int(other_indices.size()) + 3;

								// const bool is_interface = mesh_nodes.is_interface(center.index);
								// const int face_id = is_interface ? mesh_nodes.face_from_node_id(center.index) : -1;

								base.global().resize(k);

								base.global()[0].index = center.index;
								base.global()[0].val = (4. - k) / k;
								base.global()[0].node = center.node;

								base.global()[1].index = el1.index;
								base.global()[1].val = (4. - k) / k;
								base.global()[1].node = el1.node;

								base.global()[2].index = el2.index;
								base.global()[2].val = (4. - k) / k;
								base.global()[2].node = el2.node;

								// if(is_interface){
								// poly_face_to_data[face_id].local_indices.push_back(local_index);
								// }

								for (std::size_t n = 0; n < other_indices.size(); ++n)
								{
									base.global()[3 + n].index = other_indices[n];
									base.global()[3 + n].val = 4. / k;
									base.global()[3 + n].node = mesh_nodes.node_position(other_indices[n]);
								}

								const QuadraticBSpline3d spline(h_knots[x], v_knots[y], w_knots[z]);

								b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
								b.bases[local_index].set_grad([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
							}
						}
					}
				}
			}

			void create_q2_nodes(const Mesh3D &mesh, const int el_index, std::set<int> &vertex_id, std::set<int> &edge_id, std::set<int> &face_id, ElementBases &b, std::vector<LocalBoundary> &local_boundary, int &n_bases)
			{
				b.bases.resize(27);

				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
				mesh.to_face_functions(to_face);

				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> to_edge;
				mesh.to_edge_functions(to_edge);

				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> to_vertex;
				mesh.to_vertex_functions(to_vertex);

				LocalBoundary lb(el_index, BoundaryType::QUAD);

				const Navigation3D::Index start_index = mesh.get_index_from_element(el_index);
				for (int j = 0; j < 8; ++j)
				{
					const Navigation3D::Index index = to_vertex[j](start_index);
					// const int loc_index = LagrangeBasis3d::quadr_hex_face_local_nodes(mesh, index)[0];
					const int loc_index = LagrangeBasis3d::hex_face_local_nodes(false, 2, mesh, index)[0];

					int current_vertex_node_id = -1;
					Eigen::MatrixXd current_vertex_node;

					// if the edge/vertex is boundary the it is a Q2 edge
					bool is_vertex_q2 = true;

					std::vector<int> vertex_neighs;
					mesh.get_vertex_elements_neighs(index.vertex, vertex_neighs);

					for (size_t i = 0; i < vertex_neighs.size(); ++i)
					{
						if (mesh.is_spline_compatible(vertex_neighs[i]))
						{
							is_vertex_q2 = false;
							break;
						}
					}
					const bool is_vertex_boundary = mesh.is_boundary_vertex(index.vertex);

					if (is_vertex_q2)
					{
						const bool is_new_vertex = vertex_id.insert(index.vertex).second;

						if (is_new_vertex)
						{
							current_vertex_node_id = n_bases++;
							current_vertex_node = mesh.point(index.vertex);

							// if(is_vertex_boundary)//mesh.is_vertex_boundary(index.vertex))
							// bounday_nodes.push_back(current_vertex_node_id);
						}
					}

					// init new Q2 nodes
					if (current_vertex_node_id >= 0)
						b.bases[loc_index].init(2, current_vertex_node_id, loc_index, current_vertex_node);

					b.bases[loc_index].set_basis([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_basis_value_3d(2, loc_index, uv, val); });
					b.bases[loc_index].set_grad([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_grad_basis_value_3d(2, loc_index, uv, val); });
				}

				for (int j = 0; j < 12; ++j)
				{
					Navigation3D::Index index = to_edge[j](start_index);

					int current_edge_node_id = -1;
					Eigen::Matrix<double, 1, 3> current_edge_node;
					// const int loc_index = LagrangeBasis3d::quadr_hex_face_local_nodes(mesh, index)[1];
					const int loc_index = LagrangeBasis3d::hex_face_local_nodes(false, 2, mesh, index)[4];

					bool is_edge_q2 = true;

					std::vector<int> edge_neighs;
					mesh.get_edge_elements_neighs(index.edge, edge_neighs);

					for (size_t i = 0; i < edge_neighs.size(); ++i)
					{
						if (mesh.is_spline_compatible(edge_neighs[i]))
						{
							is_edge_q2 = false;
							break;
						}
					}
					const bool is_edge_boundary = mesh.is_boundary_edge(index.edge);

					if (is_edge_q2)
					{
						const bool is_new_edge = edge_id.insert(index.edge).second;

						if (is_new_edge)
						{
							current_edge_node_id = n_bases++;
							current_edge_node = mesh.edge_barycenter(index.edge);

							// if(is_edge_boundary)
							// bounday_nodes.push_back(current_edge_node_id);
						}
					}

					// init new Q2 nodes
					if (current_edge_node_id >= 0)
						b.bases[loc_index].init(2, current_edge_node_id, loc_index, current_edge_node);

					b.bases[loc_index].set_basis([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_basis_value_3d(2, loc_index, uv, val); });
					b.bases[loc_index].set_grad([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_grad_basis_value_3d(2, loc_index, uv, val); });
				}

				for (int j = 0; j < 6; ++j)
				{
					Navigation3D::Index index = to_face[j](start_index);

					int current_face_node_id = -1;

					Eigen::Matrix<double, 1, 3> current_face_node;
					const int opposite_element = mesh.switch_element(index).element;
					const bool is_face_q2 = opposite_element < 0 || !mesh.is_spline_compatible(opposite_element);
					// const int loc_index = LagrangeBasis3d::quadr_hex_face_local_nodes(mesh, index)[8];
					const int loc_index = LagrangeBasis3d::hex_face_local_nodes(false, 2, mesh, index)[8];

					if (is_face_q2)
					{
						const bool is_new_face = face_id.insert(index.face).second;

						if (is_new_face)
						{
							current_face_node_id = n_bases++;
							current_face_node = mesh.face_barycenter(index.face);

							const int b_index = loc_index - 20;

							if (opposite_element < 0) // && mesh.n_element_faces(opposite_element) == 6 && mesh.n_element_vertices(opposite_element) == 8)
							{
								// bounday_nodes.push_back(current_face_node_id);
								lb.add_boundary_primitive(index.face, b_index);
							}
						}
					}

					// init new Q2 nodes
					if (current_face_node_id >= 0)
						b.bases[loc_index].init(2, current_face_node_id, loc_index, current_face_node);

					b.bases[loc_index].set_basis([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_basis_value_3d(2, loc_index, uv, val); });
					b.bases[loc_index].set_grad([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_grad_basis_value_3d(2, loc_index, uv, val); });
				}

				// //central node always present
				b.bases[26].init(2, n_bases++, 26, mesh.cell_barycenter(el_index));
				b.bases[26].set_basis([](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_basis_value_3d(2, 26, uv, val); });
				b.bases[26].set_grad([](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_grad_basis_value_3d(2, 26, uv, val); });

				if (!lb.empty())
					local_boundary.emplace_back(lb);
			}

			void insert_into_global(const int el_index, const Local2Global &data, std::vector<Local2Global> &vec, const int size)
			{
				// ignore small weights
				if (fabs(data.val) < 1e-10)
					return;

				bool found = false;

				for (int i = 0; i < size; ++i)
				{
					if (vec[i].index == data.index)
					{
						// assert(fabs(vec[i].val - data.val) < 1e-10);
						assert((vec[i].node - data.node).norm() < 1e-10);
						found = true;
						break;
					}
				}

				if (!found)
					vec.push_back(data);
			}

			void assign_q2_weights(const Mesh3D &mesh, const int el_index, std::vector<ElementBases> &bases)
			{
				// Eigen::MatrixXd eval_p;
				std::vector<AssemblyValues> eval_p;
				const Navigation3D::Index start_index = mesh.get_index_from_element(el_index);
				ElementBases &b = bases[el_index];

				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
				mesh.to_face_functions(to_face);
				for (int f = 0; f < 6; ++f)
				{
					const Navigation3D::Index index = to_face[f](start_index);
					const int opposite_element = mesh.switch_element(index).element;

					if (opposite_element < 0 || !mesh.is_cube(opposite_element))
						continue;

					// const auto &param_p     = LagrangeBasis3d::quadr_hex_face_local_nodes_coordinates(mesh, mesh.switch_element(index));
					Eigen::Matrix<double, 9, 3> param_p;
					{
						Eigen::MatrixXd hex_loc_nodes;
						polyfem::autogen::q_nodes_3d(2, hex_loc_nodes);
						const auto opposite_indices = LagrangeBasis3d::hex_face_local_nodes(false, 2, mesh, mesh.switch_element(index));
						for (int k = 0; k < 9; ++k)
							param_p.row(k) = hex_loc_nodes.row(opposite_indices[k]);
					}
					const auto &other_bases = bases[opposite_element];

					// const auto &indices     = LagrangeBasis3d::quadr_hex_face_local_nodes(mesh, index);
					const auto &indices = LagrangeBasis3d::hex_face_local_nodes(false, 2, mesh, index);

					std::array<int, 9> sizes;

					for (int l = 0; l < 9; ++l)
						sizes[l] = b.bases[indices[l]].global().size();

					other_bases.evaluate_bases(param_p, eval_p);
					for (std::size_t i = 0; i < other_bases.bases.size(); ++i)
					{
						const auto &other_b = other_bases.bases[i];

						if (other_b.global().empty())
							continue;

						// other_b.basis(param_p, eval_p);
						assert(eval_p[i].val.size() == 9);

						// basis i of element opposite element is zero on this elements
						if (eval_p[i].val.cwiseAbs().maxCoeff() <= 1e-10)
							continue;

						for (std::size_t k = 0; k < other_b.global().size(); ++k)
						{
							for (int l = 0; l < 9; ++l)
							{
								Local2Global glob = other_b.global()[k];
								glob.val *= eval_p[i].val(l);

								insert_into_global(el_index, glob, b.bases[indices[l]].global(), sizes[l]);
							}
						}
					}
				}
			}

			void setup_data_for_polygons(const Mesh3D &mesh, const int el_index, const ElementBases &b, std::map<int, InterfaceData> &poly_face_to_data)
			{
				const Navigation3D::Index start_index = mesh.get_index_from_element(el_index);
				std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
				mesh.to_face_functions(to_face);
				for (int f = 0; f < 6; ++f)
				{
					const Navigation3D::Index index = to_face[f](start_index);

					const int opposite_element = mesh.switch_element(index).element;
					const bool is_neigh_poly = opposite_element >= 0 && mesh.is_polytope(opposite_element);

					if (is_neigh_poly)
					{
						// auto e2l = LagrangeBasis3d::quadr_hex_face_local_nodes(mesh, index);
						auto e2l = LagrangeBasis3d::hex_face_local_nodes(false, 2, mesh, index);

						InterfaceData &data = poly_face_to_data[index.face];

						for (int kk = 0; kk < e2l.size(); ++kk)
						{
							const auto idx = e2l(kk);
							data.local_indices.push_back(idx);
						}
					}
				}
			}
		} // namespace

		int SplineBasis3d::build_bases(const Mesh3D &mesh,
									   const std::string &assembler,
									   const int quadrature_order, const int mass_quadrature_order, std::vector<ElementBases> &bases, std::vector<LocalBoundary> &local_boundary, std::map<int, InterfaceData> &poly_face_to_data)
		{
			using std::max;
			assert(mesh.is_volume());

			MeshNodes mesh_nodes(mesh, true, true, 1, 1, 1);

			const int n_els = mesh.n_elements();
			bases.resize(n_els);
			local_boundary.clear();

			// bounday_nodes.clear();

			// HexQuadrature hex_quadrature;

			std::array<std::array<double, 4>, 3> h_knots;
			std::array<std::array<double, 4>, 3> v_knots;
			std::array<std::array<double, 4>, 3> w_knots;

			for (int e = 0; e < n_els; ++e)
			{
				if (!mesh.is_spline_compatible(e))
					continue;

				SpaceMatrix space;

				build_local_space(mesh, mesh_nodes, e, space, local_boundary, poly_face_to_data);

				ElementBases &b = bases[e];
				const int real_order = quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler, 2, AssemblerUtils::BasisType::SPLINE, 3);
				const int real_mass_order = mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", 2, AssemblerUtils::BasisType::SPLINE, 3);

				b.set_quadrature([real_order](Quadrature &quad) {
					HexQuadrature hex_quadrature;
					hex_quadrature.get_quadrature(real_order, quad);
				});
				b.set_mass_quadrature([real_mass_order](Quadrature &quad) {
					HexQuadrature hex_quadrature;
					hex_quadrature.get_quadrature(real_mass_order, quad);
				});
				// hex_quadrature.get_quadrature(quadrature_order, b.quadrature);
				b.bases.resize(27);

				b.set_local_node_from_primitive_func([e](const int primitive_id, const Mesh &mesh) {
					const auto &mesh3d = dynamic_cast<const Mesh3D &>(mesh);

					std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
					mesh3d.to_face_functions(to_face);

					auto start_index = mesh3d.get_index_from_element(e);
					auto index = start_index;

					int lf;
					for (lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
					{
						index = to_face[lf](start_index);
						if (index.face == primitive_id)
							break;
					}
					assert(index.face == primitive_id);

					static constexpr std::array<std::array<int, 9>, 6> face_to_index = {{
						{{2 * 9 + 0 * 3 + 0, 2 * 9 + 1 * 3 + 0, 2 * 9 + 2 * 3 + 0, 2 * 9 + 0 * 3 + 1, 2 * 9 + 1 * 3 + 1, 2 * 9 + 2 * 3 + 1, 2 * 9 + 0 * 3 + 2, 2 * 9 + 1 * 3 + 2, 2 * 9 + 2 * 3 + 2}}, // 0
						{{0 * 9 + 0 * 3 + 0, 0 * 9 + 1 * 3 + 0, 0 * 9 + 2 * 3 + 0, 0 * 9 + 0 * 3 + 1, 0 * 9 + 1 * 3 + 1, 0 * 9 + 2 * 3 + 1, 0 * 9 + 0 * 3 + 2, 0 * 9 + 1 * 3 + 2, 0 * 9 + 2 * 3 + 2}}, // 1

						{{0 * 9 + 0 * 3 + 2, 0 * 9 + 1 * 3 + 2, 0 * 9 + 2 * 3 + 2, 1 * 9 + 0 * 3 + 2, 1 * 9 + 1 * 3 + 2, 1 * 9 + 2 * 3 + 2, 2 * 9 + 0 * 3 + 2, 2 * 9 + 1 * 3 + 2, 2 * 9 + 2 * 3 + 2}}, // 2
						{{0 * 9 + 0 * 3 + 0, 0 * 9 + 1 * 3 + 0, 0 * 9 + 2 * 3 + 0, 1 * 9 + 0 * 3 + 0, 1 * 9 + 1 * 3 + 0, 1 * 9 + 2 * 3 + 0, 2 * 9 + 0 * 3 + 0, 2 * 9 + 1 * 3 + 0, 2 * 9 + 2 * 3 + 0}}, // 3

						{{0 * 9 + 2 * 3 + 0, 0 * 9 + 2 * 3 + 1, 0 * 9 + 2 * 3 + 2, 1 * 9 + 2 * 3 + 0, 1 * 9 + 2 * 3 + 1, 1 * 9 + 2 * 3 + 2, 2 * 9 + 2 * 3 + 0, 2 * 9 + 2 * 3 + 1, 2 * 9 + 2 * 3 + 2}}, // 4
						{{0 * 9 + 0 * 3 + 0, 0 * 9 + 0 * 3 + 1, 0 * 9 + 0 * 3 + 2, 1 * 9 + 0 * 3 + 0, 1 * 9 + 0 * 3 + 1, 1 * 9 + 0 * 3 + 2, 2 * 9 + 0 * 3 + 0, 2 * 9 + 0 * 3 + 1, 2 * 9 + 0 * 3 + 2}}, // 5
					}};

					Eigen::VectorXi res(9);

					for (int i = 0; i < 9; ++i)
						res(i) = face_to_index[lf][i];

					return res;
				});

				setup_knots_vectors(mesh_nodes, space, h_knots, v_knots, w_knots);
				// print_local_space(space);

				basis_for_regular_hex(mesh_nodes, space, h_knots, v_knots, w_knots, b);
				basis_for_irregulard_hex(e, mesh, mesh_nodes, space, h_knots, v_knots, w_knots, b, poly_face_to_data);
			}

			int n_bases = mesh_nodes.n_nodes();

			std::set<int> face_id;
			std::set<int> edge_id;
			std::set<int> vertex_id;

			for (int e = 0; e < n_els; ++e)
			{
				if (mesh.is_polytope(e) || mesh.is_spline_compatible(e))
					continue;

				ElementBases &b = bases[e];

				const int real_order = quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler, 2, AssemblerUtils::BasisType::CUBE_LAGRANGE, 3);
				const int real_mass_order = mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", 2, AssemblerUtils::BasisType::CUBE_LAGRANGE, 3);

				// hex_quadrature.get_quadrature(quadrature_order, b.quadrature);
				b.set_quadrature([real_order](Quadrature &quad) {
					HexQuadrature hex_quadrature;
					hex_quadrature.get_quadrature(real_order, quad);
				});
				b.set_mass_quadrature([real_mass_order](Quadrature &quad) {
					HexQuadrature hex_quadrature;
					hex_quadrature.get_quadrature(real_mass_order, quad);
				});

				b.set_local_node_from_primitive_func([e](const int primitive_id, const Mesh &mesh) {
					const auto &mesh3d = dynamic_cast<const Mesh3D &>(mesh);
					Navigation3D::Index index;

					for (int lf = 0; lf < 6; ++lf)
					{
						index = mesh3d.get_index_from_element(e, lf, 0);
						if (index.face == primitive_id)
							break;
					}
					assert(index.face == primitive_id);

					// const auto indices = LagrangeBasis3d::quadr_hex_face_local_nodes(mesh3d, index);
					const auto indices = LagrangeBasis3d::hex_face_local_nodes(false, 2, mesh3d, index);
					Eigen::VectorXi res(indices.size());

					for (size_t i = 0; i < indices.size(); ++i)
						res(i) = indices[i];

					return res;
				});

				create_q2_nodes(mesh, e, vertex_id, edge_id, face_id, b, local_boundary, n_bases);
			}

			bool missing_bases = false;
			do
			{
				missing_bases = false;
				for (int e = 0; e < n_els; ++e)
				{
					if (mesh.is_polytope(e) || mesh.is_spline_compatible(e))
						continue;

					auto &b = bases[e];
					if (b.is_complete())
						continue;

					assign_q2_weights(mesh, e, bases);

					missing_bases = missing_bases || b.is_complete();
				}
			} while (missing_bases);

			for (int e = 0; e < n_els; ++e)
			{
				if (mesh.is_polytope(e) || mesh.is_spline_compatible(e))
					continue;

				const ElementBases &b = bases[e];
				setup_data_for_polygons(mesh, e, b, poly_face_to_data);
			}

			// for(int e = 0; e < n_els; ++e)
			// {
			//     if(!mesh.is_polytope(e))
			//         continue;

			//     for (int lf = 0; lf < mesh.n_cell_faces(e); ++lf)
			//     {
			//         auto index = mesh.get_index_from_element(e, lf, 0);
			//         auto index2 = mesh.switch_element(index);
			//         if (index2.element >= 0) {
			//             auto &array = poly_face_to_data[index.face].local_indices;
			//             auto &b = bases[index2.element];
			//             array.resize(b.bases.size());
			//             std::iota(array.begin(), array.end(), 0);
			//         }
			//     }
			// }

			for (auto &k : poly_face_to_data)
			{
				auto &array = k.second.local_indices;
				std::sort(array.begin(), array.end());
				auto it = std::unique(array.begin(), array.end());
				array.resize(std::distance(array.begin(), it));
			}

			return n_bases;
		}

		void SplineBasis3d::fit_nodes(const Mesh3D &mesh, const int n_bases, std::vector<ElementBases> &gbases)
		{
			assert(false);
			// const int dim = 3;
			// const int n_constraints =  27;
			// const int n_elements = mesh.n_elements();

			// std::vector< Eigen::Triplet<double> > entries, entries_t;

			// MeshNodes nodes(mesh, true, true, 1, 1, 1);
			// // Eigen::MatrixXd tmp;
			// std::vector<AssemblyValues> tmp_val;

			// Eigen::MatrixXd node_rhs(n_constraints*n_elements, dim);
			// Eigen::MatrixXd samples(n_constraints, dim);

			// for(int i = 0; i < n_constraints; ++i)
			//     samples.row(i) = LagrangeBasis3d::quadr_hex_local_node_coordinates(i);

			// for(int i = 0; i < n_elements; ++i)
			// {
			//     auto &base = gbases[i];

			//     if(!mesh.is_cube(i))
			//         continue;

			//     auto global_ids = LagrangeBasis3d::quadr_hex_local_to_global(mesh, i);
			//     assert(global_ids.size() == n_constraints);

			//     for(int j = 0; j < n_constraints; ++j)
			//     {
			//         auto n_id = nodes.node_id_from_primitive(global_ids[j]);
			//         auto n = nodes.node_position(n_id);
			//         for(int d = 0; d < dim; ++d)
			//             node_rhs(n_constraints*i + j, d) = n(d);
			//     }

			//     base.evaluate_bases(samples, tmp_val);
			//     const auto &lbs = base.bases;

			//     const int n_local_bases = int(lbs.size());
			//     for(int j = 0; j < n_local_bases; ++j)
			//     {
			//         const Basis &b = lbs[j];
			//         const auto &tmp = tmp_val[j].val;

			//         for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			//         {
			//             for (long k = 0; k < tmp.size(); ++k)
			//             {
			//                 entries.emplace_back(n_constraints*i + k, b.global()[ii].index, tmp(k)*b.global()[ii].val);
			//                 entries_t.emplace_back(b.global()[ii].index, n_constraints*i + k, tmp(k)*b.global()[ii].val);
			//             }
			//         }
			//     }
			// }

			// Eigen::MatrixXd new_nodes(n_bases, dim);
			// {
			//     StiffnessMatrix mat(n_constraints*n_elements, n_bases);
			//     StiffnessMatrix mat_t(n_bases, n_constraints*n_elements);

			//     mat.setFromTriplets(entries.begin(), entries.end());
			//     mat_t.setFromTriplets(entries_t.begin(), entries_t.end());

			//     StiffnessMatrix A = mat_t * mat;
			//     Eigen::MatrixXd b = mat_t * node_rhs;

			//     json params = {
			//         {"mtype", -2}, // matrix type for Pardiso (2 = SPD)
			//         // {"max_iter", 0}, // for iterative solvers
			//         // {"tolerance", 1e-9}, // for iterative solvers
			//     };
			//     auto solver = LinearSolver::create("", "");
			//     solver->setParameters(params);
			//     solver->analyzePattern(A);
			//     solver->factorize(A);

			//     for(int d = 0; d < dim; ++d)
			//         solver->solve(b.col(d), new_nodes.col(d));
			// }

			// for(int i = 0; i < n_elements; ++i)
			// {
			//     auto &base = gbases[i];

			//     if(!mesh.is_cube(i))
			//         continue;

			//     auto &lbs = base.bases;
			//     const int n_local_bases = int(lbs.size());
			//     for(int j = 0; j < n_local_bases; ++j)
			//     {
			//         Basis &b = lbs[j];

			//         for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			//         {
			//             // if(nodes.is_primitive_boundary(b.global()[ii].index))
			//                 // continue;

			//             for(int d = 0; d < dim; ++d)
			//             {
			//                 b.global()[ii].node(d) = new_nodes(b.global()[ii].index, d);
			//             }
			//         }
			//     }
			// }
		}
	} // namespace basis
} // namespace polyfem
