#pragma once

#include <polyfem/Mesh.hpp>

#include <Eigen/Dense>

namespace polyfem
{
	class BoundarySampler
	{
	public:
		static void sample_parametric_quad_edge(int index, int n_samples, Eigen::MatrixXd &samples);
		static void sample_parametric_tri_edge(int index, int n_samples, Eigen::MatrixXd &samples);

		static void sample_parametric_quad_face(int index, int n_samples, Eigen::MatrixXd &samples);
		static void sample_parametric_tri_face(int index, int n_samples, Eigen::MatrixXd &samples);

		static void quadrature_for_quad_edge(int index, int order, const int gid, const Mesh &mesh, Eigen::MatrixXd &points, Eigen::VectorXd &weights);
		static void quadrature_for_tri_edge(int index, int order, const int gid, const Mesh &mesh, Eigen::MatrixXd &points, Eigen::VectorXd &weights);

		static void quadrature_for_quad_face(int index, int order, const int gid, const Mesh &mesh, Eigen::MatrixXd &points, Eigen::VectorXd &weights);
		static void quadrature_for_tri_face(int index, int order, const int gid, const Mesh &mesh, Eigen::MatrixXd &points, Eigen::VectorXd &weights);
	};
}

