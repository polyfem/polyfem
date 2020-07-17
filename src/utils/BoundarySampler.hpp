#pragma once

#include <polyfem/Mesh.hpp>

#include <Eigen/Dense>

namespace polyfem
{
	class BoundarySampler
	{
	public:
		static void sample_parametric_quad_edge(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);
		static void sample_parametric_tri_edge(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);

		static void sample_parametric_quad_face(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);
		static void sample_parametric_tri_face(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);

		static void sample_polygon_edge(int face_id, int edge_id, int n_samples, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);

		static void quadrature_for_quad_edge(int index, int order, const int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);
		static void quadrature_for_tri_edge(int index, int order, const int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);

		static void quadrature_for_quad_face(int index, int order, const int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);
		static void quadrature_for_tri_face(int index, int order, const int gid, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);

		static void quadrature_for_polygon_edge(int face_id, int edge_id, int order, const Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);

		static void normal_for_quad_edge(int index, Eigen::MatrixXd &normal);
		static void normal_for_tri_edge(int index, Eigen::MatrixXd &normal);

		static void normal_for_quad_face(int index, Eigen::MatrixXd &normal);
		static void normal_for_tri_face(int index, Eigen::MatrixXd &normal);

		static void normal_for_polygon_edge(int face_id, int edge_id, const Mesh &mesh, Eigen::MatrixXd &normal);
	};
}

