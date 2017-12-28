#include "FEBasis3d.hpp"

#include "HexQuadrature.hpp"

#include <cassert>
#include <array>

namespace poly_fem
{

	namespace
	{
		void hex_basis_basis(const int disc_order, const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
		{
			auto x=xne.col(0).array();
			auto n=xne.col(1).array();
			auto e=xne.col(2).array();

			switch(disc_order)
			{
				case 1:
				{
					switch(local_index)
					{
						case 0: val = (1-x)*(1-n)*(1-e); break;
						case 1: val = (  x)*(1-n)*(1-e); break;
						case 2: val = (  x)*(  n)*(1-e); break;
						case 3: val = (1-x)*(  n)*(1-e); break;

						case 4: val = (1-x)*(1-n)*(  e); break;
						case 5: val = (  x)*(1-n)*(  e); break;
						case 6: val = (  x)*(  n)*(  e); break;
						case 7: val = (1-x)*(  n)*(  e); break;
						default: assert(false);
					}

					break;
				}

			//No H2 implemented
				default: assert(false);
			}
		}

		void hex_basis_grad(const int disc_order, const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
		{
			val.resize(xne.rows(), 3);

			auto x=xne.col(0).array();
			auto n=xne.col(1).array();
			auto e=xne.col(2).array();

			switch(disc_order)
			{
				case 1:
				{
					switch(local_index)
					{
						case 0:
						{
						//(1-x)*(1-n)*(1-e);
							val.col(0) = -      (1-n)*(1-e);
							val.col(1) = -(1-x)      *(1-e);
							val.col(2) = -(1-x)*(1-n);

							break;
						}
						case 1:
						{
						//(  x)*(1-n)*(1-e)
							val.col(0) =        (1-n)*(1-e);
							val.col(1) = -(  x)      *(1-e);
							val.col(2) = -(  x)*(1-n);

							break;
						}
						case 2:
						{
						//(  x)*(  n)*(1-e)
							val.col(0) =        (  n)*(1-e);
							val.col(1) =  (  x)      *(1-e);
							val.col(2) = -(  x)*(  n);

							break;
						}
						case 3:
						{
						//(1-x)*(  n)*(1-e);
							val.col(0) = -       (  n)*(1-e);
							val.col(1) =  (1-x)       *(1-e);
							val.col(2) = -(1-x)*(  n);

							break;
						}
						case 4:
						{
						//(1-x)*(1-n)*(  e);
							val.col(0) = -      (1-n)*(  e);
							val.col(1) = -(1-x)      *(  e);
							val.col(2) =  (1-x)*(1-n);

							break;
						}
						case 5:
						{
						//(  x)*(1-n)*(  e);
							val.col(0) =        (1-n)*(  e);
							val.col(1) = -(  x)      *(  e);
							val.col(2) =  (  x)*(1-n);

							break;
						}
						case 6:
						{
						//(  x)*(  n)*(  e);
							val.col(0) =       (  n)*(  e);
							val.col(1) = (  x)      *(  e);
							val.col(2) = (  x)*(  n);

							break;
						}
						case 7:
						{
						//(1-x)*(  n)*(  e);
							val.col(0) = -      (  n)*(  e);
							val.col(1) =  (1-x)      *(  e);
							val.col(2) =  (1-x)*(  n);

							break;
						}

						default: assert(false);
					}

					break;
				}

				default: assert(false);
			}
		}
	}

	int FEBasis3d::build_bases(const Mesh3D &mesh, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes)
	{
		bounday_nodes.clear();

		assert(mesh.is_volume());

		const int discr_order = 1;

		bases.resize(mesh.n_elements());
		local_boundary.resize(mesh.n_elements());

		const int n_bases = int(mesh.n_pts());

		Eigen::MatrixXd node;

		HexQuadrature hex_quadrature;

		std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> to_vertex;
		to_vertex[0]= [mesh](Navigation3D::Index idx) { return idx; };
		to_vertex[1]= [mesh](Navigation3D::Index idx) { return mesh.switch_vertex(idx); };
		to_vertex[2]= [mesh](Navigation3D::Index idx) { return mesh.switch_vertex(mesh.switch_edge(mesh.switch_vertex(idx))); };
		to_vertex[3]= [mesh](Navigation3D::Index idx) { return mesh.switch_vertex(mesh.switch_edge(idx)); };

		to_vertex[4]= [mesh](Navigation3D::Index idx) { return mesh.switch_vertex(mesh.switch_edge(mesh.switch_face(idx))); };
		to_vertex[5]= [mesh](Navigation3D::Index idx) { return mesh.switch_vertex(mesh.switch_edge(mesh.switch_vertex(mesh.switch_edge(mesh.switch_face(idx))))); };
		to_vertex[6]= [mesh](Navigation3D::Index idx) { return mesh.switch_vertex(mesh.switch_edge(mesh.switch_face(mesh.switch_vertex(mesh.switch_edge(mesh.switch_vertex(idx)))))); };
		to_vertex[7]= [mesh](Navigation3D::Index idx) { return mesh.switch_vertex(mesh.switch_edge(mesh.switch_face(mesh.switch_vertex(mesh.switch_edge(idx))))); };


		std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
		to_face[0]= [mesh](Navigation3D::Index idx) { return mesh.switch_face(mesh.switch_edge(mesh.switch_vertex(mesh.switch_edge(mesh.switch_face(idx))))); };
		to_face[1]= [mesh](Navigation3D::Index idx) { return idx; };

		to_face[2]= [mesh](Navigation3D::Index idx) { return mesh.switch_face(mesh.switch_edge(mesh.switch_vertex(idx))); };
		to_face[3]= [mesh](Navigation3D::Index idx) { return mesh.switch_face(mesh.switch_edge(idx)); };

		to_face[4]= [mesh](Navigation3D::Index idx) { return mesh.switch_face(mesh.switch_edge(mesh.switch_vertex(mesh.switch_edge(mesh.switch_vertex(idx))))); };
		to_face[5]= [mesh](Navigation3D::Index idx) { return mesh.switch_face(idx); };

		for(int e = 0; e < mesh.n_elements(); ++e)
		{
			const int n_el_vertices = mesh.n_element_vertices(e);
			const int n_el_faces = mesh.n_element_faces(e);
			ElementBases &b=bases[e];

			if(n_el_vertices == 8 && n_el_faces == 6)
			{
				hex_quadrature.get_quadrature(quadrature_order, b.quadrature);
				b.bases.resize(n_el_vertices);

				for (int i = 0; i < n_el_faces; ++i)
				{
					Navigation3D::Index index = mesh.get_index_from_element(e);
					index = to_face[i](index);
					// std::cout<<i<<" "<<index.face<<" "<<e<<std::endl;
					if (mesh.switch_element(index).element < 0)
					{

						switch(i)
						{
							case 0: local_boundary[e].set_top_edge_id(index.face); local_boundary[e].set_top_boundary(); break;
							case 1: local_boundary[e].set_bottom_edge_id(index.face); local_boundary[e].set_bottom_boundary(); break;

							case 2: local_boundary[e].set_right_edge_id(index.face); local_boundary[e].set_right_boundary(); break;
							case 3: local_boundary[e].set_left_edge_id(index.face); local_boundary[e].set_left_boundary(); break;

							case 4: local_boundary[e].set_front_edge_id(index.face); local_boundary[e].set_front_boundary(); break;
							case 5: local_boundary[e].set_back_edge_id(index.face); local_boundary[e].set_back_boundary(); break;
						}
					}
				}

				for (int j = 0; j < n_el_vertices; ++j)
				{
					Navigation3D::Index index = mesh.get_index_from_element(e);
					index = to_vertex[j](index);
					const int global_index = index.vertex;

					// std::cout<<j<<" "<<global_index<<" "<<e<<std::endl;

					if (mesh.is_boundary_vertex(global_index)) {
						bounday_nodes.push_back(global_index);
					}

					mesh.point(global_index, node);
						// std::cout<<node<<std::endl;
					b.bases[j].init(global_index, j, node);

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { hex_basis_basis(discr_order, j, uv, val); });
					b.bases[j].set_grad( [discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  hex_basis_grad(discr_order, j, uv, val); });
				}
					// std::cout<<std::endl;
			}
			else
			{
				assert(false);
				//TODO triangulate element and implement tets...
			}
		}

		std::sort(bounday_nodes.begin(), bounday_nodes.end());
		auto it = std::unique(bounday_nodes.begin(), bounday_nodes.end());
		bounday_nodes.resize(std::distance(bounday_nodes.begin(), it));

		return n_bases;
	}
}
