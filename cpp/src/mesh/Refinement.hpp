#pragma once

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

} // namespace poly_fem
