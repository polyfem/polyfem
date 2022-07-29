#pragma once

#include <Eigen/Core>
#ifdef POLYFEM_WITH_CLIPPER
#include <clipper.hpp>
#endif

namespace polyfem::utils
{
	/// @brief Multiplicative scale factor when converting from double to integer coordinates.
	static constexpr int64_t DOUBLE_TO_INT_SCALE_FACTOR = 1l << 51;

#ifdef POLYFEM_WITH_CLIPPER

	class Point
	{
	public:
		Point() = delete;

		static ClipperLib::IntPoint toClipper(const Eigen::RowVector2d &p);
		static Eigen::RowVector2d fromClipper(const ClipperLib::IntPoint &p);
	};

	class Polygon
	{
	public:
		Polygon() = delete;

		static ClipperLib::Path toClipper(const Eigen::MatrixXd &V);
		static Eigen::MatrixXd fromClipper(const ClipperLib::Path &path);

		static std::vector<Eigen::MatrixXd> clip(
			const Eigen::MatrixXd &subject_polygon,
			const Eigen::MatrixXd &clipping_polygon);
	};

#endif

} // namespace polyfem::utils