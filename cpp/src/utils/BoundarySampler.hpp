#pragma once

#include "FEBasis2d.hpp"
#include "FEBasis3d.hpp"

namespace poly_fem
{
	class BoundarySampler
	{
	public:
		static void sample_parametric_quad_edge(int index, int n_samples, Eigen::MatrixXd &samples)
		{
			auto endpoints = FEBasis2d::quad_local_node_coordinates_from_edge(index);
			const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
			samples.resize(n_samples, endpoints.cols());

			for (int c = 0; c < 2; ++c) {
				samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
			}
		}

		static void sample_parametric_tri_edge(int index, int n_samples, Eigen::MatrixXd &samples)
		{
			auto endpoints = FEBasis2d::tri_local_node_coordinates_from_edge(index);
			const Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);
			samples.resize(n_samples, endpoints.cols());

			for (int c = 0; c < 2; ++c) {
				samples.col(c) = (1.0 - t.array()).matrix() * endpoints(0, c) + t * endpoints(1, c);
			}
		}

		static void sample_parametric_quad_face(int index, int n_samples, Eigen::MatrixXd &samples)
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

		static void sample_parametric_tri_face(int index, int n_samples, Eigen::MatrixXd &samples)
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
	};
}

