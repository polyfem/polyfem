#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

#include <Eigen/Dense>

namespace polyfem
{
	namespace utils
	{
		class BoundarySampler
		{
		public:
			static void sample_parametric_quad_edge(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);
			static void sample_parametric_tri_edge(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);

			static void sample_parametric_quad_face(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);
			static void sample_parametric_tri_face(int index, int n_samples, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);

			static void sample_polygon_edge(int face_id, int edge_id, int n_samples, const mesh::Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples);

			static void quadrature_for_quad_edge(int index, int order, const int gid, const mesh::Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);
			static void quadrature_for_tri_edge(int index, int order, const int gid, const mesh::Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);

			static void quadrature_for_quad_face(int index, int order, const int gid, const mesh::Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);
			static void quadrature_for_tri_face(int index, int order, const int gid, const mesh::Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);

			static void quadrature_for_polygon_edge(int face_id, int edge_id, int order, const mesh::Mesh &mesh, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::VectorXd &weights);

			static void normal_for_quad_edge(int index, Eigen::MatrixXd &normal);
			static void normal_for_tri_edge(int index, Eigen::MatrixXd &normal);

			static void normal_for_quad_face(int index, Eigen::MatrixXd &normal);
			static void normal_for_tri_face(int index, Eigen::MatrixXd &normal);

			static void normal_for_polygon_edge(int face_id, int edge_id, const mesh::Mesh &mesh, Eigen::MatrixXd &normal);

			static Eigen::MatrixXd tet_local_node_coordinates_from_face(int lf);
			static Eigen::MatrixXd hex_local_node_coordinates_from_face(int lf);

			//sample boundary facet, uv are local (ref) values, samples are global coordinates
			static bool sample_boundary(const mesh::LocalBoundary &local_boundary, const int n_samples, const mesh::Mesh &mesh, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &samples, Eigen::VectorXi &global_primitive_ids);
			//compute quadrature for boundary facet, uv are local (ref) values, samples are global coordinates
			static bool boundary_quadrature(const mesh::LocalBoundary &local_boundary, const int order, const mesh::Mesh &mesh, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::MatrixXd &normals, Eigen::VectorXd &weights, Eigen::VectorXi &global_primitive_ids);
			//compute quadrature for one boundary facet, uv are local (ref) values, samples are global coordinates
			static bool boundary_quadrature(const mesh::LocalBoundary &local_boundary, const int order, const mesh::Mesh &mesh, const int i, const bool skip_computation, Eigen::MatrixXd &uv, Eigen::MatrixXd &points, Eigen::MatrixXd &normal, Eigen::VectorXd &weights);
		};
	} // namespace utils
} // namespace polyfem
