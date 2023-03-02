#pragma once

#ifdef POLYFEM_WITH_CLIPPER

#include <Eigen/Core>
#include <clipper.hpp>
namespace polyfem::utils
{
	/// @brief Multiplicative scale factor when converting from double to integer coordinates.
	static constexpr int64_t DOUBLE_TO_INT_SCALE_FACTOR = 1l << 51;

	class PolygonClipping
	{
	public:
		PolygonClipping() = delete;

		/// @brief Clip a polygon using convex polygon.
		/// @param[in] subject_polygon Polygon to clip in clockwise order.
		/// @param[in] clipping_polygon Convex polygon to clip with in clockwise order.
		/// @return Clipped polygon(s).
		static std::vector<Eigen::MatrixXd> clip(
			const Eigen::MatrixXd &subject_polygon,
			const Eigen::MatrixXd &clipping_polygon);

		static ClipperLib::IntPoint toClipperPoint(const Eigen::RowVector2d &p);
		static Eigen::RowVector2d fromClipperPoint(const ClipperLib::IntPoint &p);

		static ClipperLib::Path toClipperPolygon(const Eigen::MatrixXd &V);
		static Eigen::MatrixXd fromClipperPolygon(const ClipperLib::Path &path);
	};

	class TriangleClipping
	{
	public:
		TriangleClipping() = delete;

		/// @brief Clip a triangle using triangle.
		/// @param[in] subject_triangle Triangle to clip.
		/// @param[in] clipping_triangle Triangle to clip with.
		/// @return Clipped polygon(s).
		/// @return Triangularization of the clipped (convex) polygon. Each entry is a matrix of size 3×2 containing the three vertices of each triangle.
		static std::vector<Eigen::MatrixXd> clip(
			const Eigen::MatrixXd &subject_triangle,
			const Eigen::MatrixXd &clipping_triangle);
	};

} // namespace polyfem::utils

#endif