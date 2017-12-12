#include "QuadBasis.hpp"
#include "TriBasis.hpp"

#include "QuadQuadrature.hpp"
#include "TriQuadrature.hpp"

#include <cassert>
#include <algorithm>

namespace poly_fem
{
	template<typename T>
	Eigen::MatrixXd theta1(T &t)
	{
		return (1-t) * (1-2 * t);
	}

	template<typename T>
	Eigen::MatrixXd theta2(T &t)
	{
		return t * (2 * t - 1);
	}

	template<typename T>
	Eigen::MatrixXd theta3(T &t)
	{
		return 4 * t * (1 - t);
	}



	template<typename T>
	Eigen::MatrixXd dtheta1(T &t)
	{
		return -3+4*t;
	}

	template<typename T>
	Eigen::MatrixXd dtheta2(T &t)
	{
		return -1+4*t;
	}

	template<typename T>
	Eigen::MatrixXd dtheta3(T &t)
	{
		return 4-8*t;
	}

	int QuadBasis::build_bases(const Mesh &mesh, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes)
	{
		bounday_nodes.clear();

		assert(!mesh.is_volume());

		const int discr_order = 1;

		bases.resize(mesh.n_elements());
		local_boundary.resize(mesh.n_elements());

		const int n_bases = int(mesh.n_pts());

		Eigen::MatrixXd node;

		QuadQuadrature quad_quadrature;
		TriQuadrature tri_quadrature;

		for(int e = 0; e < mesh.n_elements(); ++e)
		{
			const int n_el_vertices = mesh.n_element_vertices(e);
			ElementBases &b=bases[e];

			if(n_el_vertices == 4)
			{
				quad_quadrature.get_quadrature(quadrature_order, b.quadrature);

				b.bases.resize(n_el_vertices);

				Navigation::Index index = mesh.get_index_from_face(e);
				for (int j = 0; j < n_el_vertices; ++j)
				{
					if (mesh.switch_face(index).face < 0) {
						bounday_nodes.push_back(index.vertex);
						bounday_nodes.push_back(mesh.switch_vertex(index).vertex);

						switch(j)
						{
							case 0: local_boundary[e].set_top_edge_id(index.edge); local_boundary[e].set_top_boundary(); break;
							case 1: local_boundary[e].set_left_edge_id(index.edge); local_boundary[e].set_left_boundary(); break;
							case 2: local_boundary[e].set_bottom_edge_id(index.edge); local_boundary[e].set_bottom_boundary(); break;
							case 3: local_boundary[e].set_right_edge_id(index.edge); local_boundary[e].set_right_boundary(); break;
						}
					}

					const int global_index = index.vertex;

					mesh.point(global_index, node);
					b.bases[j].init(global_index, j, node);

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { QuadBasis::basis(discr_order, j, uv, val); });
					b.bases[j].set_grad( [discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  QuadBasis::grad(discr_order, j, uv, val); });

					index = mesh.next_around_face(index);
				}
			}
			else if(n_el_vertices == 3)
			{
				tri_quadrature.get_quadrature(quadrature_order, b.quadrature);

				b.bases.resize(n_el_vertices);

				Navigation::Index index = mesh.get_index_from_face(e);
				for (int j = 0; j < n_el_vertices; ++j) {
					if (mesh.switch_face(index).face < 0) {
						bounday_nodes.push_back(index.vertex);
						bounday_nodes.push_back(mesh.switch_vertex(index).vertex);

						switch(j)
						{
							case 0: local_boundary[e].set_right_edge_id(index.edge); local_boundary[e].set_right_boundary(); break;
							case 1: local_boundary[e].set_bottom_edge_id(index.edge); local_boundary[e].set_bottom_boundary(); break;
							case 2: local_boundary[e].set_left_edge_id(index.edge); local_boundary[e].set_left_boundary(); break;
						}
					}

					const int global_index = index.vertex;

					mesh.point(global_index, node);
					b.bases[j].init(global_index, j, node);

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { TriBasis::basis(discr_order, j, uv, val); });
					b.bases[j].set_grad( [discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  TriBasis::grad(discr_order, j, uv, val); });

					index = mesh.next_around_face(index);
				}
			}
			else
			{
				assert(false);
				//TODO triangulate element...
			}
		}

		std::sort(bounday_nodes.begin(), bounday_nodes.end());
		auto it = std::unique(bounday_nodes.begin(), bounday_nodes.end());
		bounday_nodes.resize(std::distance(bounday_nodes.begin(), it));

		return n_bases;
	}


	void QuadBasis::basis(const int discr_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
	{
		switch(discr_order)
		{
			case 1:
			{
				switch(local_index)
				{
					case 0: val = (1-uv.col(0).array())*(1-uv.col(1).array()); break;
					case 1: val = uv.col(0).array()*(1-uv.col(1).array()); break;
					case 2: val = uv.col(0).array()*uv.col(1).array(); break;
					case 3: val = (1-uv.col(0).array())*uv.col(1).array(); break;
					default: assert(false);
				}

				break;
			}

			case 2:
			{
				auto &u = uv.col(0).array();
				auto &v = uv.col(1).array();

				switch(local_index)
				{
					case 0: val = theta1(u).array() * theta1(v).array(); break;
					case 1: val = theta2(u).array() * theta1(v).array(); break;
					case 2: val = theta2(u).array() * theta2(v).array(); break;
					case 3: val = theta1(u).array() * theta2(v).array(); break;
					case 4: val = theta3(u).array() * theta1(v).array(); break;
					case 5: val = theta2(u).array() * theta3(v).array(); break;
					case 6: val = theta3(u).array() * theta2(v).array(); break;
					case 7: val = theta1(u).array() * theta3(v).array(); break;
					case 8: val = theta3(u).array() * theta3(v).array(); break;
					default: assert(false);
				}

				break;
			}
			//No Q3 implemented
			default: assert(false);
		}
	}

	void QuadBasis::grad(const int discr_order, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
	{
		val.resize(uv.rows(),2);

		switch(discr_order)
		{
			case 1:
			{
				switch(local_index)
				{
					case 0:
					{
						val.col(0) = -(1-uv.col(1).array());
						val.col(1) = -(1-uv.col(0).array());

						break;
					}
					case 1:
					{
						val.col(0) = (1-uv.col(1).array());
						val.col(1) =   -uv.col(0);

						break;
					}
					case 2:
					{
						val.col(0) = uv.col(1);
						val.col(1) = uv.col(0);

						break;
					}
					case 3:
					{
						val.col(0) = -uv.col(1);
						val.col(1) = 1-uv.col(0).array();

						break;
					}

					default: assert(false);
				}

				break;
			}

			case 2:
			{
				auto u=uv.col(0).array();
				auto v=uv.col(1).array();

				switch(local_index)
				{
					case 0:
					{
						val.col(0) = dtheta1(u).array() * theta1(v).array();
						val.col(1) = theta1(u).array() * dtheta1(v).array();
						break;
					}
					case 1:
					{
						val.col(0) = dtheta2(u).array() * theta1(v).array();
						val.col(1) = theta2(u).array() * dtheta1(v).array();
						break;
					}
					case 2:
					{
						val.col(0) = dtheta2(u).array() * theta2(v).array();
						val.col(1) = theta2(u).array() * dtheta2(v).array();
						break;
					}
					case 3:
					{
						val.col(0) = dtheta1(u).array() * theta2(v).array();
						val.col(1) = theta1(u).array() * dtheta2(v).array();
						break;
					}
					case 4:
					{
						val.col(0) = dtheta3(u).array() * theta1(v).array();
						val.col(1) = theta3(u).array() * dtheta1(v).array();
						break;
					}
					case 5:
					{
						val.col(0) = dtheta2(u).array() * theta3(v).array();
						val.col(1) = theta2(u).array() * dtheta3(v).array();
						break;
					}
					case 6:
					{
						val.col(0) = dtheta3(u).array() * theta2(v).array();
						val.col(1) = theta3(u).array() * dtheta2(v).array();
						break;
					}
					case 7:
					{
						val.col(0) = dtheta1(u).array() * theta3(v).array();
						val.col(1) = theta1(u).array() * dtheta3(v).array();
						break;
					}
					case 8:
					{
						val.col(0) = dtheta3(u).array() * theta3(v).array();
						val.col(1) = theta3(u).array() * dtheta3(v).array();
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
