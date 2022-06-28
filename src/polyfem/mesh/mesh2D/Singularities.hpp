#pragma once

#include <geogram/mesh/mesh.h>
#include <Eigen/Dense>
#include <vector>

namespace polyfem
{

	namespace mesh
	{
		//
		// Compute singular vertices in a polygonal mesh.
		//
		// @param[in]  M                Input surface mesh
		// @param[out] V                List of singular vertices
		// @param[in]  regular_degree  { Degree of regular vertices (3 for triangle
		//                             meshes, 4 for quad mesh)
		// @param[in]  ignore_border    Mark vertices on the border as well
		//
		void singular_vertices(const GEO::Mesh &M, Eigen::VectorXi &V, int regular_degree = 4, bool ignore_border = true);

		//
		// Compute singular edges in a polygonal mesh.
		//
		// @param[in]  M      Input surface mesh
		// @param[in]  V      List of singular vertices
		// @param[out] E      List of edges connecting singular vertices
		//
		void singular_edges(const GEO::Mesh &M, const Eigen::VectorXi &V, Eigen::MatrixX2i &E);

		//
		// Compute singularity graph in a polygonal mesh.
		//
		// @param[in]  M                Input surface mesh
		// @param[out] V                List of singular vertices
		// @param[out] E                List of edges connecting singular vertices
		// @param[in]  regular_degree  { Degree of regular vertices (3 for triangle
		//                             meshes, 4 for quad mesh)
		// @param[in]  ignore_border    Mark vertices on the border as well
		//
		void singularity_graph(const GEO::Mesh &M, Eigen::VectorXi &V, Eigen::MatrixX2i &E, int regular_degree = 4, bool ignore_border = true);

		//
		// Creates polygonal patches around singularities specified in the form of a
		// graph.
		//
		// @param[in,out] M     { Surface mesh to modify }
		// @param[in]     V      List of singular vertices
		// @param[in]     E      List of singular edges
		// @param[in]     t     { Interpolation parameter to place the subdivided
		//                      vertices around the singularities (between 0 and 1) }
		//
		void create_patch_around_singularities(GEO::Mesh &M, const Eigen::VectorXi &V, const Eigen::MatrixX2i &E, double t = 0.5);
	} // namespace mesh
} // namespace polyfem
