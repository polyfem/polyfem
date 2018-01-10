#pragma once

#include "Mesh.hpp"
#include <geogram/mesh/mesh.h>
#include <Eigen/Dense>
#include <vector>


namespace poly_fem
{

///
/// Retrieve a 3D vector with the position of a given vertex. Contrary to
/// geogram's version, this function works with both single and double precision
/// meshes, as well as 2D meshes too.
///
/// @param[in]  M     { Input mesh }
/// @param[in]  v     { Vertex index whose position to retrieve }
///
/// @return     { Position of the given vertex in 3D }
///
GEO::vec3 mesh_vertex(const GEO::Mesh &M, GEO::index_t v);

// Compute facet barycenter.
//
// @param[in]  M     { Input mesh }
// @param[in]  f     { Facet whose barycenter to compute }
//
// @return     { Barycenter position in 3D }
//
GEO::vec3 facet_barycenter(const GEO::Mesh &M, GEO::index_t f);

// Create a new mesh vertex with given coordinates.
//
// @param      M     { Mesh to modify }
// @param[in]  p     { New vertex position }
//
// @return     { Index of the newly created vertex }
//
GEO::index_t mesh_create_vertex(GEO::Mesh &M, const GEO::vec3 &p);

///
/// @brief      Compute the type of each facet in a surface mesh.
///
/// @param[in]  M             { Input surface mesh }
/// @param[out] element_tags  { Types of each facet element }
///
void compute_element_tags(const GEO::Mesh &M, std::vector<ElementType> &element_tags);

///
/// @brief         Orient facets of a 2D mesh so that each connected component
///                has positive volume
///
/// @param[in,out] M     { Surface mesh to reorient }
///
void orient_normals_2d(GEO::Mesh &M);

///
/// @brief         Reorder vertices of a mesh using color tags, so that vertices are ordered by
///                increasing colors
///
/// @param[in,out] V     { #V x d input mesh vertices }
/// @param[in,out] F     { #F x k input mesh faces }
/// @param[in]     C     { #V per vertex color tag }
/// @param[out]    R     { max(C)+1 vector of starting indices for each colors (last value is the
///                      total number of vertices) }
///
void reorder_mesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const Eigen::VectorXi &C, Eigen::VectorXi &R);

} // namespace poly_fem
