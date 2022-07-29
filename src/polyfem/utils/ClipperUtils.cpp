#include "ClipperUtils.hpp"

namespace polyfem::utils
{
#ifdef POLYFEM_WITH_CLIPPER

	namespace
	{
		double scale_double_coord(double coord)
		{
			return coord * (double)DOUBLE_TO_INT_SCALE_FACTOR;
		}
	} // namespace

	ClipperLib::IntPoint Point::toClipper(const Eigen::RowVector2d &p)
	{
		ClipperLib::IntPoint r;
		assert(abs(scale_double_coord(p.x())) <= ClipperLib::hiRange);
		r.X = (ClipperLib::cInt)std::round(scale_double_coord(p.x()));
		assert(abs(scale_double_coord(p.y())) <= ClipperLib::hiRange);
		r.Y = (ClipperLib::cInt)std::round(scale_double_coord(p.y()));
		return r;
	}

	Eigen::RowVector2d Point::fromClipper(const ClipperLib::IntPoint &p)
	{
		return Eigen::RowVector2d(p.X, p.Y) / DOUBLE_TO_INT_SCALE_FACTOR;
	}

	ClipperLib::Path Polygon::toClipper(const Eigen::MatrixXd &V)
	{
		ClipperLib::Path path(V.rows());
		for (size_t i = 0; i < path.size(); ++i)
		{
			path[i] = Point::toClipper(V.row(i));
		}
		return path;
	}

	Eigen::MatrixXd Polygon::fromClipper(const ClipperLib::Path &path)
	{
		Eigen::MatrixXd V(path.size(), 2);
		for (size_t i = 0; i < path.size(); ++i)
		{
			V.row(i) = Point::fromClipper(path[i]);
		}
		return V;
	}

	std::vector<Eigen::MatrixXd> Polygon::clip(
		const Eigen::MatrixXd &subject_polygon,
		const Eigen::MatrixXd &clipping_polygon)
	{
		using namespace ClipperLib;

		Clipper clipper;

		clipper.AddPath(Polygon::toClipper(subject_polygon), ptSubject, /*closed=*/true);
		clipper.AddPath(Polygon::toClipper(clipping_polygon), ptClip, /*closed=*/true);

		Paths solution;
		clipper.Execute(ctIntersection, solution, pftNonZero, pftNonZero);

		std::vector<Eigen::MatrixXd> result;
		result.reserve(solution.size());
		for (const auto &path : solution)
		{
			result.push_back(Polygon::fromClipper(path));
		}

		return result;
	}

#endif
} // namespace polyfem::utils