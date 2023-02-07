#include "ClipperUtils.hpp"

#ifdef POLYFEM_WITH_CLIPPER

#include <polyfem/utils/GeometryUtils.hpp>

namespace polyfem::utils
{
	std::vector<Eigen::MatrixXd> PolygonClipping::clip(
		const Eigen::MatrixXd &subject_polygon,
		const Eigen::MatrixXd &clipping_polygon)
	{
		using namespace ClipperLib;

		Clipper clipper;

		clipper.AddPath(toClipperPolygon(subject_polygon), ptSubject, /*closed=*/true);
		clipper.AddPath(toClipperPolygon(clipping_polygon), ptClip, /*closed=*/true);

		Paths solution;
		clipper.Execute(ctIntersection, solution, pftNonZero, pftNonZero);

		std::vector<Eigen::MatrixXd> result;
		result.reserve(solution.size());
		for (const auto &path : solution)
		{
			result.push_back(fromClipperPolygon(path));
		}

		return result;
	}

	std::vector<Eigen::MatrixXd> TriangleClipping::clip(
		const Eigen::MatrixXd &subject_triangle,
		const Eigen::MatrixXd &clipping_triangle)
	{
		const std::vector<Eigen::MatrixXd> overlap =
			PolygonClipping::clip(
				triangle_to_clockwise_order(subject_triangle),
				triangle_to_clockwise_order(clipping_triangle));
		assert(overlap.size() <= 1);

		if (overlap.empty() || overlap[0].rows() < 3)
			return std::vector<Eigen::MatrixXd>();

		return triangle_fan(overlap[0]);
	}

	namespace
	{
		double scale_double_coord(double coord)
		{
			return coord * (double)DOUBLE_TO_INT_SCALE_FACTOR;
		}
	} // namespace

	ClipperLib::IntPoint PolygonClipping::toClipperPoint(const Eigen::RowVector2d &p)
	{
		ClipperLib::IntPoint r;
		assert(abs(scale_double_coord(p.x())) <= ClipperLib::hiRange);
		r.X = (ClipperLib::cInt)std::round(scale_double_coord(p.x()));
		assert(abs(scale_double_coord(p.y())) <= ClipperLib::hiRange);
		r.Y = (ClipperLib::cInt)std::round(scale_double_coord(p.y()));
		return r;
	}

	Eigen::RowVector2d PolygonClipping::fromClipperPoint(const ClipperLib::IntPoint &p)
	{
		return Eigen::RowVector2d(p.X, p.Y) / DOUBLE_TO_INT_SCALE_FACTOR;
	}

	ClipperLib::Path PolygonClipping::toClipperPolygon(const Eigen::MatrixXd &V)
	{
		ClipperLib::Path path(V.rows());
		for (size_t i = 0; i < path.size(); ++i)
		{
			path[i] = toClipperPoint(V.row(i));
		}
		return path;
	}

	Eigen::MatrixXd PolygonClipping::fromClipperPolygon(const ClipperLib::Path &path)
	{
		Eigen::MatrixXd V(path.size(), 2);
		for (size_t i = 0; i < path.size(); ++i)
		{
			V.row(i) = fromClipperPoint(path[i]);
		}
		return V;
	}
} // namespace polyfem::utils

#endif