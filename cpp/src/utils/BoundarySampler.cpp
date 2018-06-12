#include "BoundarySampler.hpp"

#include "FEBasis2d.hpp"
#include "FEBasis3d.hpp"

#include "LineQuadrature.hpp"
#include "TriQuadrature.hpp"
#include "QuadQuadrature.hpp"

namespace poly_fem
{
	void BoundarySampler::quadrature_for_quad_edge(int index, int order, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		auto endpoints = FEBasis2d::tri_local_node_coordinates_from_edge(index);

		Quadrature quad;
		LineQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		points.resize(quad.points.rows(), endpoints.cols());

		for (int c = 0; c < 2; ++c) {
			points.col(c) = (1.0 - quad.points.array()) * endpoints(0, c) + quad.points.array() * endpoints(1, c);
		}

		weights = quad.weights;
	}

	void BoundarySampler::quadrature_for_tri_edge(int index, int order, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		auto endpoints = FEBasis2d::tri_local_node_coordinates_from_edge(index);

		Quadrature quad;
		LineQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		points.resize(quad.points.rows(), endpoints.cols());

		for (int c = 0; c < 2; ++c) {
			points.col(c) = (1.0 - quad.points.array()) * endpoints(0, c) + quad.points.array() * endpoints(1, c);
		}

		weights = quad.weights;
	}

	void BoundarySampler::quadrature_for_quad_face(int index, int order, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		auto endpoints = FEBasis3d::hex_local_node_coordinates_from_face(index);

		Quadrature quad;
		QuadQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		const int n_pts = quad.points.rows();
		points.resize(n_pts, endpoints.cols());

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
	}

	void BoundarySampler::quadrature_for_tri_face(int index, int order, Eigen::MatrixXd &points, Eigen::VectorXd &weights)
	{
		auto endpoints = FEBasis3d::tet_local_node_coordinates_from_face(index);
		Quadrature quad;
		TriQuadrature quad_rule; quad_rule.get_quadrature(order, quad);

		const int n_pts = quad.points.rows();
		points.resize(n_pts, endpoints.cols());

		for (int i = 0; i < n_pts; ++i) {
			const double b1 = quad.points(i, 0);
			const double b3 = quad.points(i, 1);
			const double b2 = 1 - b1 - b3;

			for (int c = 0; c < 3; ++c) {
				points(i, c) = b1*endpoints(0, c) + b2 * endpoints(1, c) + b3 * endpoints(2, c);
			}
		}

		weights = quad.weights;
	}



	void BoundarySampler::sample_parametric_quad_edge(int index, int n_samples, Eigen::MatrixXd &samples)
	{
		auto endpoints = FEBasis2d::quad_local_node_coordinates_from_edge(index);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples, endpoints.cols());

		for (int c = 0; c < 2; ++c) {
			samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
		}
	}

	void BoundarySampler::sample_parametric_tri_edge(int index, int n_samples, Eigen::MatrixXd &samples)
	{
		auto endpoints = FEBasis2d::tri_local_node_coordinates_from_edge(index);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples, endpoints.cols());

		for (int c = 0; c < 2; ++c) {
			samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
		}
	}

	void BoundarySampler::sample_parametric_quad_face(int index, int n_samples, Eigen::MatrixXd &samples)
	{
		auto endpoints = FEBasis3d::hex_local_node_coordinates_from_face(index);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples*n_samples, endpoints.cols());
		Eigen::MatrixXd left(n_samples, endpoints.cols());
		Eigen::MatrixXd right(n_samples, endpoints.cols());

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

	void BoundarySampler::sample_parametric_tri_face(int index, int n_samples, Eigen::MatrixXd &samples)
	{
		auto endpoints = FEBasis3d::tet_local_node_coordinates_from_face(index);
		const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
		samples.resize(n_samples*n_samples, endpoints.cols());

		int counter = 0;
		for (int u = 0; u < n_samples; ++u) {
			for (int v = 0; v < n_samples; ++v) {
				if(t(u) + t(v) > 1) continue;

				for (int c = 0; c < 3; ++c){
					samples(counter, c) = t(u)*endpoints(0,c) + t(v)*endpoints(1,c) + (1-t(u)-t(v))*endpoints(2,c);
				}
				++counter;
			}
		}
		samples.conservativeResize(counter, 3);
	}
}

