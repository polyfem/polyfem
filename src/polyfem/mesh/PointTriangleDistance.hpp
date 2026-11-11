#pragma once

namespace polyfem::mesh
{
	///
	/// @brief      Squared distance between a point and a triangle.
	///
	/// David Eberly's algorithm (https://www.geometrictools.com), with the exact
	/// expressions of geogram's Geom::point_triangle_squared_distance, which
	/// PolyFEM used to call through geogram's facet AABB. Like geogram, this file
	/// is compiled without floating-point contraction (see CMakeLists.txt), so
	/// the results are identical to the former ones.
	///
	/// @param[in]  p           Query point (3 coordinates)
	/// @param[in]  v0, v1, v2  Triangle vertices (3 coordinates each)
	///
	/// @return     Squared distance
	///
	double point_triangle_squared_distance(const double *p, const double *v0, const double *v1, const double *v2);
} // namespace polyfem::mesh
