#include "HexBasis.hpp"

#include "HexQuadrature.hpp"

#include <cassert>

namespace poly_fem
{

	int HexBasis::build_bases(const Mesh3D &mesh, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes)
	{
		// for(long i=0; i < mesh.els.rows(); ++i)
		// {
		// 	std::vector<Basis> &b=bases[i];
		// 	b.resize(8);

		// 	for(int j = 0; j < 8; ++j)
		// 	{
		// 		const int global_index = mesh.els(i,j);
		// 		b[j].init(global_index, j, mesh.pts.row(global_index));

		// 		b[j].set_basis([disc_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { HexBasis::basis(disc_order, j, uv, val); });
		// 		b[j].set_grad( [disc_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  HexBasis::grad(disc_order, j, uv, val); });
		// 	}
		// }

		// return n_bases;



		bounday_nodes.clear();

		assert(mesh.is_volume());

		const int discr_order = 1;

		bases.resize(mesh.n_elements());
		local_boundary.resize(mesh.n_elements());

		const int n_bases = int(mesh.n_pts());

		Eigen::MatrixXd node;

		HexQuadrature hex_quadrature;

		for(int e = 0; e < mesh.n_elements(); ++e)
		{
			const int n_el_vertices = mesh.n_element_vertices(e);
			ElementBases &b=bases[e];

			if(n_el_vertices == 8)
			{
				hex_quadrature.get_quadrature(quadrature_order, b.quadrature);

				b.bases.resize(n_el_vertices);

				Navigation3D::Index index = mesh.get_index_from_element_face(e);
				for (int j = 0; j < n_el_vertices; ++j)
				{
					if (mesh.switch_element(index).element < 0) {
						bounday_nodes.push_back(index.vertex);
						bounday_nodes.push_back(mesh.switch_vertex(index).vertex);

						switch(j)
						{
							case 1: local_boundary[e].set_top_edge_id(index.edge); local_boundary[e].set_top_boundary(); break;
							case 2: local_boundary[e].set_left_edge_id(index.edge); local_boundary[e].set_left_boundary(); break;
							case 3: local_boundary[e].set_bottom_edge_id(index.edge); local_boundary[e].set_bottom_boundary(); break;
							case 0: local_boundary[e].set_right_edge_id(index.edge); local_boundary[e].set_right_boundary(); break;
						}
					}

					const int global_index = index.vertex;

					mesh.point(global_index, node);
					b.bases[j].init(global_index, j, node);

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { HexBasis::basis(discr_order, j, uv, val); });
					b.bases[j].set_grad( [discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  HexBasis::grad(discr_order, j, uv, val); });

					index = mesh.next_around_element(index);
				}
			}
			else if(n_el_vertices == 3)
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

	void HexBasis::basis(const int disc_order, const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
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

	void HexBasis::grad(const int disc_order, const int local_index, const Eigen::MatrixXd &xne, Eigen::MatrixXd &val)
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
						val.col(1) =  (1-x)      *(1-e);
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
