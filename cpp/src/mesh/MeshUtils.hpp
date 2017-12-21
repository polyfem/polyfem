#pragma once

#include "Mesh.hpp"
#include <geogram/mesh/mesh.h>
#include <Eigen/Dense>
#include <vector>


namespace poly_fem
{

///
/// @brief      Compute the type of each facet in a surface mesh.
///
/// @param[in]  M             { Input surface mesh }
/// @param[out] element_tags  { Types of each facet element }
///
void compute_element_tags(const GEO::Mesh &M, std::vector<ElementType> &element_tags);

} // namespace poly_fem
