#pragma once

#include <Eigen/Dense>
#include <vector>


namespace poly_fem
{

///
/// Clip a polygon by a half-plane.
/// https://github.com/alicevision/geogram/blob/cfbc0a5827d71d59f8bcf0369cc1731ef12f82ef/src/examples/graphics/demo_Delaunay2d/main.cpp#L677
///
/// @param[in]  P       { Input polygon }
/// @param[in]  q1      { First endpoint of the clipping line }
/// @param[in]  q2      { Second endpoint of the clipping line }
/// @param[out] result  { Clipped polygon }
///
void clip_polygon_by_half_plane(const Eigen::MatrixXd &P, const Eigen::RowVector2d &q1,
	const Eigen::RowVector2d &q2, Eigen::MatrixXd &result);

///
/// Determine the kernel of the given polygon.
///
/// @param[in]  IV    { #IV x (2|3) vertex positions around the input polygon }
/// @param[out] OV    { #OV x (2|3) vertex positions around the output polygon }
///
void compute_visibility_kernel(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV);

///
/// Determine whether a polygon is star-shaped or not.
///
/// @param[in]  IV    { #IV x (2|3) of vertex positions around the polygon }
/// @param[out] bary  { The barycenter of the kernel }
///
/// @return     True if star shaped, False otherwise.
///
bool is_star_shaped(const Eigen::MatrixXd &IV, Eigen::RowVector3d &bary);

} // namespace poly_fem
