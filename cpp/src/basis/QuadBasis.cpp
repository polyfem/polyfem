#include "QuadBasis.hpp"
#include "TriBasis.hpp"

#include "QuadQuadrature.hpp"
#include "TriQuadrature.hpp"

#include <cassert>
#include <algorithm>

namespace poly_fem
{
	namespace
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

		struct BasisLocValue
		{
			Eigen::MatrixXd node;
			int global_index;
		};

		struct BasisValue
		{
			std::vector<BasisLocValue> values;
			int n_el_vertices;
			int n_bases;
		};


		int prepare_q1(const Mesh2D &mesh, std::vector< BasisValue > &values, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes)
		{
			Eigen::MatrixXd node;
			values.resize(mesh.n_elements());

			for(int e = 0; e < mesh.n_elements(); ++e)
			{
				BasisValue &bv = values[e];
				const int n_el_vertices = mesh.n_element_vertices(e);
				bv.n_el_vertices = n_el_vertices;
				bv.n_bases = n_el_vertices;

				std::vector<BasisLocValue> &lvals = bv.values;
				lvals.resize(bv.n_bases);

				if(n_el_vertices == 4)
				{

					Navigation::Index index = mesh.get_index_from_face(e);
					for (int j = 0; j < n_el_vertices; ++j)
					{
						if (mesh.switch_face(index).face < 0) {
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

						lvals[j].global_index = index.vertex;
						mesh.point(index.vertex, lvals[j].node);

						index = mesh.next_around_face(index);
					}
				}
				else if(n_el_vertices == 3)
				{

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

						lvals[j].global_index = index.vertex;
						mesh.point(index.vertex, lvals[j].node);

						index = mesh.next_around_face(index);
					}
				}
				else
				{
					assert(false);
				//TODO triangulate element...
				}

			}

			return mesh.n_pts();
		}


		int prepare_q2(const Mesh2D &mesh, std::vector< BasisValue > &values, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes)
		{
			Eigen::MatrixXd node;
			values.resize(mesh.n_elements());

			const int n_vertex_nodes = mesh.n_pts();
			const int n_edge_nodes = mesh.n_edges();
			const int n_face_nodes = mesh.n_elements();

			std::vector<Eigen::Matrix<double, 1, 2> > edge_nodes(mesh.n_edges());

			for(int e = 0; e < mesh.n_edges(); ++e)
			{
				edge_nodes[e] = mesh.edge_mid_point(e);
			}

			for(int e = 0; e < mesh.n_elements(); ++e)
			{
				BasisValue &bv = values[e];
				const int n_el_vertices = mesh.n_element_vertices(e);
				bv.n_el_vertices = n_el_vertices;
				bv.n_bases = 9;

				std::vector<BasisLocValue> &lvals = bv.values;
				lvals.resize(bv.n_bases);

				if(n_el_vertices == 4)
				{

					Navigation::Index index = mesh.get_index_from_face(e);
					for (int j = 0; j < n_el_vertices; ++j)
					{
						if (mesh.switch_face(index).face < 0) {
							bounday_nodes.push_back(index.vertex);
							bounday_nodes.push_back(index.edge + n_vertex_nodes);
							bounday_nodes.push_back(mesh.switch_vertex(index).vertex);

							switch(j)
							{
								case 0: local_boundary[e].set_top_edge_id(index.edge); local_boundary[e].set_top_boundary(); break;
								case 1: local_boundary[e].set_left_edge_id(index.edge); local_boundary[e].set_left_boundary(); break;
								case 2: local_boundary[e].set_bottom_edge_id(index.edge); local_boundary[e].set_bottom_boundary(); break;
								case 3: local_boundary[e].set_right_edge_id(index.edge); local_boundary[e].set_right_boundary(); break;
							}
						}

						lvals[2*j].global_index = index.vertex;
						mesh.point(index.vertex, lvals[2*j].node);

						lvals[2*j+1].global_index = index.edge + n_vertex_nodes;
						lvals[2*j+1].node = edge_nodes[index.edge];

						index = mesh.next_around_face(index);
					}

					lvals[8].global_index = n_vertex_nodes + n_edge_nodes + e;
					lvals[8].node = mesh.node_from_face(e);
				}
				else
				{
					assert(false);
					//TODO triangulate element and implement p2...
				}

			}

			return n_vertex_nodes + n_edge_nodes + n_face_nodes;
		}
	}

	int QuadBasis::build_bases(const Mesh2D &mesh, const int quadrature_order, const int discr_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes)
	{
		bounday_nodes.clear();

		assert(!mesh.is_volume());

		bases.resize(mesh.n_elements());
		local_boundary.resize(mesh.n_elements());

		std::vector< BasisValue > values;
		int n_bases;
		if(discr_order == 1)
			n_bases = prepare_q1(mesh, values, local_boundary, bounday_nodes);
		else if(discr_order == 2)
			n_bases = prepare_q2(mesh, values, local_boundary, bounday_nodes);
		else
			assert(false);


		QuadQuadrature quad_quadrature;
		TriQuadrature tri_quadrature;

		for(int e = 0; e < mesh.n_elements(); ++e)
		{
			const BasisValue &bv = values[e];
			ElementBases &b=bases[e];
			const int n_el_vertices = bv.n_el_vertices;
			const int n_loc_bases = bv.n_bases;
			b.bases.resize(n_loc_bases);

			if(n_el_vertices == 4)
			{
				quad_quadrature.get_quadrature(quadrature_order, b.quadrature);

				for (int j = 0; j < n_loc_bases; ++j)
				{
					const BasisLocValue &lbv = bv.values[j];
					const int global_index = lbv.global_index;

					b.bases[j].init(global_index, j, lbv.node);

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { QuadBasis::basis(discr_order, j, uv, val); });
					b.bases[j].set_grad( [discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  QuadBasis::grad(discr_order, j, uv, val); });
				}
			}
			else if(n_el_vertices == 3)
			{
				tri_quadrature.get_quadrature(quadrature_order, b.quadrature);

				for (int j = 0; j < n_loc_bases; ++j)
				{
					const BasisLocValue &lbv = bv.values[j];
					const int global_index = lbv.global_index;

					b.bases[j].init(global_index, j, lbv.node);

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { TriBasis::basis(discr_order, j, uv, val); });
					b.bases[j].set_grad( [discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  TriBasis::grad(discr_order, j, uv, val); });
				}
			}
			else
			{
				assert(false);
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
				auto &u = uv.col(1).array();
				auto &v = uv.col(0).array();

				switch(local_index)
				{

					case 0: val = u*(1-v); break;
					case 1: val = (1-u)*(1-v); break;
					case 2: val = (1-u)*v; break;
					case 3: val = u*v; break;
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
					case 2: val = theta2(u).array() * theta1(v).array(); break;
					case 4: val = theta2(u).array() * theta2(v).array(); break;
					case 6: val = theta1(u).array() * theta2(v).array(); break;
					case 1: val = theta3(u).array() * theta1(v).array(); break;
					case 3: val = theta2(u).array() * theta3(v).array(); break;
					case 5: val = theta3(u).array() * theta2(v).array(); break;
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
				auto &u = 1-uv.col(1).array();
				auto &v = uv.col(0).array();

				switch(local_index)
				{
					case 0:
					{
						val.col(0) = (1-v);
						val.col(1) = -u;

						break;
					}
					case 1:
					{
						val.col(0) = -(1-v);
						val.col(1) = -(1-u);

						break;
					}
					case 2:
					{
						val.col(0) = -v;
						val.col(1) = (1-u);

						break;
					}
					case 3:
					{
						val.col(0) = v;
						val.col(1) = u;

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
					case 2:
					{
						val.col(0) = dtheta2(u).array() * theta1(v).array();
						val.col(1) = theta2(u).array() * dtheta1(v).array();
						break;
					}
					case 4:
					{
						val.col(0) = dtheta2(u).array() * theta2(v).array();
						val.col(1) = theta2(u).array() * dtheta2(v).array();
						break;
					}
					case 6:
					{
						val.col(0) = dtheta1(u).array() * theta2(v).array();
						val.col(1) = theta1(u).array() * dtheta2(v).array();
						break;
					}
					case 1:
					{
						val.col(0) = dtheta3(u).array() * theta1(v).array();
						val.col(1) = theta3(u).array() * dtheta1(v).array();
						break;
					}
					case 3:
					{
						val.col(0) = dtheta2(u).array() * theta3(v).array();
						val.col(1) = theta2(u).array() * dtheta3(v).array();
						break;
					}
					case 5:
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
