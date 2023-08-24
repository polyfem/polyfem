#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

namespace polyfem::mesh
{
	/// @brief Stitch a triangle mesh (V, F) together by removing duplicate vertices
	/// @param[in] V Input vertices
	/// @param[in] F Input faces
	/// @param[out] V_out Output vertices (duplicate vertices removed)
	/// @param[out] F_out Output faces (updated to use V_out)
	/// @param[in] epsilon Tolerance for duplicate vertices
	void stitch_mesh(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		Eigen::MatrixXd &V_out,
		Eigen::MatrixXi &F_out,
		const double epsilon = 1e-5);

	/// @brief Stitch a triangle mesh (V, F) together by removing duplicate vertices
	///
	/// Also removes weights in W that correspond to duplicate vertices.
	///
	/// @param[in] V Input vertices
	/// @param[in] F Input faces
	/// @param[in] W Input weights
	/// @param[out] V_out Output vertices (duplicate vertices removed)
	/// @param[out] F_out Output faces (updated to use V_out)
	/// @param[out] W_out Output weights (updated to use V_out)
	/// @param[in] epsilon Tolerance for duplicate vertices
	void stitch_mesh(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const std::vector<Eigen::Triplet<double>> &W,
		Eigen::MatrixXd &V_out,
		Eigen::MatrixXi &F_out,
		std::vector<Eigen::Triplet<double>> &W_out,
		const double epsilon = 1e-5);

	/// @brief Compute the maximum edge length of a triangle mesh (V, F)
	/// @param V vertices
	/// @param F triangular faces
	double max_edge_length(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

	// Regular tessellation

	/// @brief Compute the barycentric coordinates of a regular grid of triangles
	/// @param[in] n Number of triangles along each edge
	/// @param[out] V Vertices of the regular grid
	/// @param[out] F Faces of the regular grid
	void regular_grid_triangle_barycentric_coordinates(
		const int n, Eigen::MatrixXd &V, Eigen::MatrixXi &F);

	/// @brief Tessilate a triangle mesh (V, F) with regular grids of triangles of maximum edge length
	/// @param V vertices
	/// @param F triangular faces
	/// @param max_edge_length maximum edge length
	/// @param V_out tessilated vertices
	/// @param F_out tessilated faces
	void regular_grid_tessellation(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const double max_edge_length,
		Eigen::MatrixXd &V_out,
		Eigen::MatrixXi &F_out);

	// Irregular tessellation

	/// @brief Refine an edge (a, b) so each refined edge has length at most max_edge_length
	/// @param a first vertex of the edge
	/// @param b second vertex of the edge
	/// @param max_edge_length maximum edge length
	/// @return vertices of the refined edge (in order)
	Eigen::MatrixXd
	refine_edge(const VectorNd &a, const VectorNd &b, const double max_edge_length);

	/// @brief Refine the edges of a triangle (a, b, c) so each refined edge has length at most max_edge_length
	/// @param a first vertex of the triangle
	/// @param b second vertex of the triangle
	/// @param c third vertex of the triangle
	/// @param max_edge_len maximum edge length
	/// @param V vertices of the refined edges
	/// @param E refined edges
	void refine_triangle_edges(
		const VectorNd &a,
		const VectorNd &b,
		const VectorNd &c,
		const double max_edge_len,
		Eigen::MatrixXd &V,
		Eigen::MatrixXi &E);

	/// @brief Refine a triangle (a, b, c) into a well shaped triangle mesh
	/// @note Uses Triangle to perform the refinement (requires POLYFEM_WITH_TRIANGLE).
	/// @param a first vertex of the triangle
	/// @param b second vertex of the triangle
	/// @param c third vertex of the triangle
	/// @param max_edge_len maximum edge length of the refined triangle mesh
	/// @param UV barycentric coordinates of the refined triangle
	/// @param F faces of the refined triangle
	void irregular_triangle_barycentric_coordinates(
		const Eigen::Vector3d &a,
		const Eigen::Vector3d &b,
		const Eigen::Vector3d &c,
		const double max_edge_length,
		Eigen::MatrixXd &UV,
		Eigen::MatrixXi &F);

	/// @brief Refine a triangle (a, b, c) into a well shaped triangle mesh
	/// @note Uses Triangle to perform the refinement (requires POLYFEM_WITH_TRIANGLE).
	/// @param a first vertex of the triangle
	/// @param b second vertex of the triangle
	/// @param c third vertex of the triangle
	/// @param max_edge_len maximum edge length of the refined triangle mesh
	/// @param V vertices of the refined triangle
	/// @param F faces of the refined triangle
	void irregular_triangle(
		const Eigen::Vector3d &a,
		const Eigen::Vector3d &b,
		const Eigen::Vector3d &c,
		const double max_edge_length,
		Eigen::MatrixXd &V,
		Eigen::MatrixXi &F);

	/// @brief Tessilate a triangle mesh (V, F) with well shaped triangles
	/// @note Uses Triangle to perform the refinement (requires POLYFEM_WITH_TRIANGLE).
	/// @param V vertices
	/// @param F triangular faces
	/// @param max_edge_length maximum edge length
	/// @param V_out tessilated vertices
	/// @param F_out tessilated faces
	void irregular_tessellation(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const double max_edge_length,
		Eigen::MatrixXd &V_out,
		Eigen::MatrixXi &F_out);
} // namespace polyfem::mesh