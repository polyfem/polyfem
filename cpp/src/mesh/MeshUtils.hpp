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

} // namespace poly_fem
