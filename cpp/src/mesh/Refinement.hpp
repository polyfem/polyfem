#pragma once

#include <geogram/mesh/mesh.h>
#include <Eigen/Dense>
#include <vector>


namespace poly_fem
{

// Compute a graph (V,E) where V are the edges of the input quad mesh, and E
// connects edges from opposite sides of the input quads.
//
// @param[in]  Q               { #Q x 4 input quads }
// @param[out] edge_index      { Map (f, lv) -> edge index for edge (lv, lv+1) }
// @param[out] adj             { Adjacency graph }
// @param[out] pairs_of_edges  { List of mesh edges, corresponding to the
//                             vertices of the output graph }
// @param[out] pairs_of_quads  { List of adjacent quads }
// @param[out] quad_index      { Map (f, lv) -> index of the quad across edge
//                             (lv, lv+1) }
//
void edge_adjacency_graph(const Eigen::MatrixXi &Q, Eigen::MatrixXi &edge_index,
	std::vector<std::vector<int>> &adj,
	std::vector<std::pair<int, int>> *pairs_of_edges = nullptr,
	std::vector<std::pair<int, int>> *pairs_of_quads = nullptr,
	Eigen::MatrixXi *quad_index = nullptr);

// Instantiate a periodic 2D pattern (triangle-mesh) on a given quad mesh
//
// @param[in]  IV    { #IV x 3 input quad mesh vertices }
// @param[in]  IF    { #IF x 4 input quad mesh facets }
// @param[in]  PV    { #PV x (2|3) input pattern vertices in [0,1]^2 }
// @param[in]  PF    { #PF x (3|4) input pattern facets }
// @param[out] OV    { #OV x 3 output mesh vertices }
// @param[out] OF    { #OF x 3 output mesh facets }
//
// @return     { Return true in case of success. }
//
bool instanciate_pattern(
	const Eigen::MatrixXd &IV, const Eigen::MatrixXi &IF,
	const Eigen::MatrixXd &PV, const Eigen::MatrixXi &PF,
	Eigen::MatrixXd &OV, Eigen::MatrixXi &OF);

//
// Refine a quad-mesh by splitting each quad into 4 quads.
//
// @param[in]  IV    { #IV x 3 input quad mesh vertices }
// @param[in]  IF    { #IF x 4 input quad mesh facets }
// @param[out] OV    { #OV x 3 output mesh vertices }
// @param[out] OF    { #OF x 4 output mesh facets }
//
void refine_quad_mesh(const Eigen::MatrixXd &IV, const Eigen::MatrixXi &IF,
	Eigen::MatrixXd &OV, Eigen::MatrixXi &OF);

///
/// Split a polygon using polar refinement. The input polygon must be
/// star-shaped. A one-ring of quads are create on the outer ring of the
/// polygon, while at the center a new polygonal facet is created around the
/// barycenter of the kernel polygon.
///
/// @param[in]  IV    { #IV x (2|3) of vertex positions around the polygon }
/// @param[out] OV    { #OF v (2|3) output vertex positions }
/// @param[out] OF    { list of output polygonal face indices }
/// @param[in]  t     { Interpolation parameter to place the new vertices on the
///                   edge from the barycenter to the outer polygon vertices (0
///                   being at the center, 1 being at the boundary) }
///
void polar_split(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV, std::vector<std::vector<int>> &OF, double t = 0.5);

///
/// Refine a polygonal mesh. Quads and triangles are split into quads. If
/// `refine_polygons` is set to `true`, then polygonal facets are also split
/// into a layer of padding quads, and a new polygon is created around the
/// barycenter
///
/// @param[in]  M_in             { Surface mesh to subdivide }
/// @param[out] M_out            { Refined mesh }
/// @param[in]  refine_polygons  { Whether to refine polygons using polar
///                              refinement }
/// @param[in]  t                { Interpolation parameter }
///
void refine_polygonal_mesh(const GEO::Mesh &M_in, GEO::Mesh &M_out, bool refine_polygons = false, double t = 0.5);

} // namespace poly_fem
