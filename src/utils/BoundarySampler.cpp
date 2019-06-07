#include <polyfem/BoundarySampler.hpp>

#include <polyfem/LineQuadrature.hpp>
#include <polyfem/TriQuadrature.hpp>
#include <polyfem/QuadQuadrature.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <polyfem/Mesh2D.hpp>

#include <cassert>

namespace polyfem
{
	namespace
	{
		Eigen::RowVector2d quad_local_node_coordinates(int local_index) {
			assert(local_index >= 0 && local_index < 4);
			Eigen::MatrixXd p;
			polyfem::autogen::q_nodes_2d(1, p);
			return Eigen::RowVector2d(p(local_index, 0), p(local_index, 1));
		}

		Eigen::RowVector2d tri_local_node_coordinates(int local_index) {
			Eigen::MatrixXd p;
			polyfem::autogen::p_nodes_2d(1, p);
			return Eigen::RowVector2d(p(local_index, 0), p(local_index, 1));
		}

		Eigen::RowVector3d linear_tet_local_node_coordinates(int local_index) {
			Eigen::MatrixXd p;
			polyfem::autogen::p_nodes_3d(1, p);
			return Eigen::RowVector3d(p(local_index, 0), p(local_index, 1), p(local_index, 2));
		}

		Eigen::RowVector3d linear_hex_local_node_coordinates(int local_index) {
			Eigen::MatrixXd p;
			polyfem::autogen::q_nodes_3d(1, p);
			return Eigen::RowVector3d(p(local_index, 0), p(local_index, 1), p(local_index, 2));
		}


		Eigen::Matrix2d quad_local_node_coordinates_from_edge(int le)
		{
			Eigen::Matrix2d res(2,2);
			res.row(0) = quad_local_node_coordinates(le);
			res.row(1) = quad_local_node_coordinates((le+1)%4);

			return res;
		}

		Eigen::Matrix2d tri_local_node_coordinates_from_edge(int le)
		{
			Eigen::Matrix2d  res(2,2);
			res.row(0) = tri_local_node_coordinates(le);
			res.row(1) = tri_local_node_coordinates((le+1)%3);

			return res;
		}

		Eigen::MatrixXd tet_local_node_coordinates_from_face(int lf)
		{
			Eigen::Matrix<int, 4, 3> fv;
			fv.row(0) << 0, 1, 2;
			fv.row(1) << 0, 1, 3;
			fv.row(2) << 1, 2, 3;
			fv.row(3) << 2, 0, 3;

			Eigen::MatrixXd res(3,3);
			for(int i = 0; i < 3; ++i)
				res.row(i) = linear_tet_local_node_coordinates(fv(lf, i));

			return res;
		}

		Eigen::MatrixXd hex_local_node_coordinates_from_face(int lf)
		{
			Eigen::Matrix<int, 6, 4> fv;
			fv.row(0) << 0, 3, 7, 4;
			fv.row(1) << 1, 2, 6, 5;
			fv.row(2) << 0, 1, 5, 4;
			fv.row(3) << 3, 2, 6, 7;
			fv.row(4) << 0, 1, 2, 3;
			fv.row(5) << 4, 5, 6, 7;

			Eigen::MatrixXd res(4,3);
			for(int i = 0; i < 4; ++i)
				res.row(i) = linear_hex_local_node_coordinates(fv(lf, i));

			return res;
		}
	}

	void BoundarySampler::quadrature_for_quad_edge(int index, int order, int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		auto endpoints = quad_local_node_coordinates_from_edge(index);

		Quadrature quad;
		LineQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		points.resize(quad.points.rows(), endpoints.cols());
		uv.resize(quad.points.rows(), 2);

		uv.col(0) = (1.0 - quad.points.array());
		uv.col(1) = quad.points.array();

		for (int c = 0; c < 2; ++c) {
			points.col(c) = (1.0 - quad.points.array()) * endpoints(0, c) + quad.points.array() * endpoints(1, c);
		}

		weights = quad.weights;
		weights *= mesh.edge_length(gid);
	}

	void BoundarySampler::quadrature_for_tri_edge(int index, int order, int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		auto endpoints = tri_local_node_coordinates_from_edge(index);

		Quadrature quad;
		LineQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		points.resize(quad.points.rows(), endpoints.cols());
		uv.resize(quad.points.rows(), 2);

		uv.col(0) = (1.0 - quad.points.array());
		uv.col(1) = quad.points.array();

		for (int c = 0; c < 2; ++c) {
			points.col(c) = (1.0 - quad.points.array()) * endpoints(0, c) + quad.points.array() * endpoints(1, c);
		}

		weights = quad.weights;
		weights *= mesh.edge_length(gid);
	}

	void BoundarySampler::quadrature_for_quad_face(int index, int order, int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		auto endpoints = hex_local_node_coordinates_from_face(index);

		Quadrature quad;
		QuadQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		const int n_pts = quad.points.rows();
		points.resize(n_pts, endpoints.cols());

		uv.resize(quad.points.rows(), 4);
		uv.col(0) = quad.points.col(0);
		uv.col(1) = 1 - uv.col(0).array();
		uv.col(2) = quad.points.col(1);
		uv.col(3) = 1 - uv.col(2).array();

		for (int i = 0; i < n_pts; ++i) {
			const double b1 = quad.points(i, 0);
			const double b2 = 1 - b1;

			const double b3 = quad.points(i, 1);
			const double b4 = 1 - b3;

			for (int c = 0; c < 3; ++c) {
				points(i, c) = b3 * (b1 * endpoints(0, c) + b2 * endpoints(1, c)) + b4 * (b1 * endpoints(3, c) + b2 * endpoints(2, c));
			}
		}

		weights = quad.weights;
		weights *= mesh.quad_area(gid);
	}

	void BoundarySampler::quadrature_for_tri_face(int index, int order, int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		auto endpoints = tet_local_node_coordinates_from_face(index);
		Quadrature quad;
		TriQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		const int n_pts = quad.points.rows();
		points.resize(n_pts, endpoints.cols());

		uv.resize(quad.points.rows(), 3);
		uv.col(0) = quad.points.col(0);
		uv.col(1) = quad.points.col(1);
		uv.col(2) = 1 - uv.col(0).array() - uv.col(1).array();

		for (int i = 0; i < n_pts; ++i) {
			const double b1 = quad.points(i, 0);
			const double b3 = quad.points(i, 1);
			const double b2 = 1 - b1 - b3;

			for (int c = 0; c < 3; ++c) {
				points(i, c) = b1*endpoints(0, c) + b2 * endpoints(1, c) + b3 * endpoints(2, c);
			}
		}

		weights = quad.weights;
		//2* because weights sum to 1/2 already
		weights *= 2*mesh.tri_area(gid);
	}



	void BoundarySampler::sample_parametric_quad_edge(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
	{
		auto endpoints = quad_local_node_coordinates_from_edge(index);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples, endpoints.cols());

		uv.resize(n_samples, 2);

		uv.col(0) = (1.0 - t.array());
		uv.col(1) = t.array();

		for (int c = 0; c < 2; ++c) {
			samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
		}
	}

	void BoundarySampler::sample_parametric_tri_edge(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
	{
		auto endpoints = tri_local_node_coordinates_from_edge(index);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples, endpoints.cols());

		uv.resize(n_samples, 2);

		uv.col(0) = (1.0 - t.array());
		uv.col(1) = t.array();

		for (int c = 0; c < 2; ++c) {
			samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
		}
	}

	void BoundarySampler::sample_parametric_quad_face(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
	{
		auto endpoints = hex_local_node_coordinates_from_face(index);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples*n_samples, endpoints.cols());
		Eigen::MatrixXd left(n_samples, endpoints.cols());
		Eigen::MatrixXd right(n_samples, endpoints.cols());


		uv.resize(n_samples*n_samples, endpoints.cols());
		uv.setZero();

		for (int c = 0; c < 3; ++c) {
			left.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
			right.col(c) = (1.0 - t.array()).matrix() * endpoints(3, c) + t * endpoints(2, c);
		}
		for (int c = 0; c < 3; ++c) {
			Eigen::MatrixXd x = (1.0 - t.array()).matrix() * left.col(c).transpose() + t * right.col(c).transpose();
			assert(x.size() == n_samples * n_samples);

			samples.col(c) = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
		}
	}

	void BoundarySampler::sample_parametric_tri_face(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
	{
		auto endpoints = tet_local_node_coordinates_from_face(index);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples*n_samples, endpoints.cols());

		uv.resize(n_samples*n_samples, 3);

		int counter = 0;
		for (int u = 0; u < n_samples; ++u) {
			for (int v = 0; v < n_samples; ++v) {
				if(t(u) + t(v) > 1) continue;

				uv(counter, 0) = t(u);
				uv(counter, 1) = t(v);
				uv(counter, 2) = 1 - t(u) - t(v);

				for (int c = 0; c < 3; ++c){
					samples(counter, c) = t(u)*endpoints(0,c) + t(v)*endpoints(1,c) + (1-t(u)-t(v))*endpoints(2,c);
				}
				++counter;
			}
		}
		samples.conservativeResize(counter, 3);
		uv.conservativeResize(counter, 3);
	}

	void BoundarySampler::sample_polygon_edge(int face_id, int edge_id, int n_samples, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples)
	{
		assert(!mesh.is_volume());

		const Mesh2D &mesh2d = dynamic_cast<const Mesh2D&>(mesh);

		auto index = mesh2d.get_index_from_face(face_id);

		bool found = false;
		for(int i = 0; i < mesh2d.n_face_vertices(face_id); ++i)
		{
			if(index.edge == edge_id)
			{
				found = true;
				break;
			}

			index = mesh2d.next_around_face(index);
		}

		assert(found);

		auto p0 = mesh2d.point(index.vertex);
		auto p1 = mesh2d.point(mesh2d.switch_edge(index).vertex);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples, p0.cols());

		uv.resize(n_samples, 2);

		uv.col(0) = (1.0 - t.array());
		uv.col(1) = t.array();

		for (int c = 0; c < 2; ++c) {
			samples.col(c) = (1.0 - t.array()) * p0(c) + t.array() * p1(c);
		}
	}

	void BoundarySampler::quadrature_for_polygon_edge(int face_id, int edge_id, int order, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		assert(!mesh.is_volume());

		const Mesh2D &mesh2d = dynamic_cast<const Mesh2D&>(mesh);

		auto index = mesh2d.get_index_from_face(face_id);

		bool found = false;
		for(int i = 0; i < mesh2d.n_face_vertices(face_id); ++i)
		{
			if(index.edge == edge_id)
			{
				found = true;
				break;
			}

			index = mesh2d.next_around_face(index);
		}

		assert(found);

		auto p0 = mesh2d.point(index.vertex);
		auto p1 = mesh2d.point(mesh2d.switch_edge(index).vertex);
		Quadrature quad;
		LineQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		points.resize(quad.points.rows(), p0.cols());
		uv.resize(quad.points.rows(), 2);

		uv.col(0) = (1.0 - quad.points.array());
		uv.col(1) = quad.points.array();

		for (int c = 0; c < 2; ++c) {
			points.col(c) = (1.0 - quad.points.array()) * p0(c) + quad.points.array() * p1(c);
		}

		weights = quad.weights;
		weights *= mesh.edge_length(edge_id);
	}
}

