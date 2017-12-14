#pragma once


#include <geogram/mesh/mesh.h>
#include <Eigen/Dense>
#include <vector>


namespace poly_fem
{

//
// Compute singular vertices in a polygonal mesh.
//
// @param[in]  M               { Input surface mesh }
// @param[out] V               { List of singular vertices }
// @param[in]  regular_degree  { Degree of regular vertices (3 for triangle meshes, 4 for quad mesh)
//
void singular_vertices(const GEO::Mesh &M, Eigen::VectorXi &V, int regular_degree = 4);

//
// Compute singular edges in a polygonal mesh.
//
// @param[in]  M     { Input surface mesh }
// @param[in]  V     { List of singular vertices }
// @param[out] E     { List of edges connecting singular vertices }
//
void singular_edges(const GEO::Mesh &M, const Eigen::VectorXi &V, Eigen::MatrixX2i &E);

//
// Compute singularity graph in a polygonal mesh.
//
// @param[in]  M               { parameter_description }
// @param[out] V               { List of singular vertices }
// @param[out] E               { List of edges connecting singular vertices }
// @param[in]  regular_degree  { Degree of regular vertices (3 for triangle
//                             meshes, 4 for quad mesh)
//
void singularity_graph(const GEO::Mesh &M, Eigen::VectorXi &V, Eigen::MatrixX2i &E, int regular_degree = 4);

} // namespace poly_fem
