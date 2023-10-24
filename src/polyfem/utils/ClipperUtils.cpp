#include "ClipperUtils.hpp"

#include <polyclipper2d.hh>
#include <polyclipper3d.hh>

#include <polyfem/utils/GeometryUtils.hpp>

namespace polyfem::utils
{
	std::vector<Eigen::MatrixXd> PolygonClipping::clip(
		const Eigen::MatrixXd &subject_polygon,
		const Eigen::MatrixXd &clipping_polygon)
	{
		// #ifdef POLYFEM_WITH_CLIPPER
		// using namespace ClipperLib;

		// Clipper clipper;

		// clipper.AddPath(toClipperPolygon(subject_polygon), ptSubject, /*closed=*/true);
		// clipper.AddPath(toClipperPolygon(clipping_polygon), ptClip, /*closed=*/true);

		// Paths solution;
		// clipper.Execute(ctIntersection, solution, pftNonZero, pftNonZero);

		// std::vector<Eigen::MatrixXd> result;
		// result.reserve(solution.size());
		// for (const auto &path : solution)
		// {
		// 	result.push_back(fromClipperPolygon(path));
		// }

		// return result;
		// #else
		using namespace PolyClipper;

		// Convert the subject polygon to PolyClipper format.
		std::vector<Vertex2d<>> poly;
		{
			std::vector<Vector2d> positions;
			positions.reserve(subject_polygon.rows());
			std::vector<std::vector<int>> neighbors;
			neighbors.reserve(subject_polygon.rows());
			for (int i = 0; i < subject_polygon.rows(); i++)
			{
				positions.push_back(toPolyClipperVector(subject_polygon.row(i)));
				neighbors.emplace_back(2);
				// Clockwise neighbor is the next vertex (assuming the polygon is given in clockwise order).
				neighbors.back()[0] = (i + 1) % subject_polygon.rows();
				// Counter-clockwise neighbor is the previous vertex (assuming the polygon is given in clockwise order).
				neighbors.back()[1] = i == 0 ? (subject_polygon.rows() - 1) : (i - 1); // -1 % n = -1 but we want n-1;
			}

			initializePolygon(poly, positions, neighbors);
		}

		// Clip the subject polygon with the clipping polygon.
		{
			// Convert the clipping polygon to planes.
			std::vector<Plane<internal::VectorAdapter<Vector2d>>> planes;
			for (int i = 0; i < clipping_polygon.rows(); i++)
			{
				const Vector2d e0 = toPolyClipperVector(clipping_polygon.row(i));
				const Vector2d e1 = toPolyClipperVector(clipping_polygon.row((i + 1) % clipping_polygon.rows()));
				const Vector2d e = e1 - e0;
				planes.emplace_back(e0, Vector2d(e.y, -e.x).unitVector());
			}

			clipPolygon(poly, planes);
		}

		// Convert the clipped polygon to Eigen format.
		std::vector<Eigen::MatrixXd> result;
		{
			std::vector<std::vector<Eigen::Vector2d>> polygons;

			std::vector<bool> found_vertex(poly.size(), false);
			for (int i = 0; i < poly.size(); i++)
			{
				if (!found_vertex[i])
					polygons.emplace_back();

				int j = i;
				while (!found_vertex[j])
				{
					found_vertex[j] = true;
					polygons.back().emplace_back(fromPolyClipperVector(poly[j].position));
					j = poly[j].neighbors.first; // clockwise neighbor
				}
			}

			for (const std::vector<Eigen::Vector2d> &polygon : polygons)
			{
				result.emplace_back(polygon.size(), 2);
				for (int i = 0; i < polygon.size(); i++)
					result.back().row(i) = polygon[i];
			}
		}

		return result;
		// #endif
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

	std::vector<Eigen::MatrixXd> TetrahedronClipping::clip(
		const Eigen::MatrixXd &subject_tet,
		const Eigen::MatrixXd &clipping_tet)
	{
		using namespace PolyClipper;
		assert(subject_tet.rows() == 4 && subject_tet.cols() == 3);
		assert(clipping_tet.rows() == 4 && clipping_tet.cols() == 3);
		assert(tetrahedron_volume(subject_tet) > 0);
		assert(tetrahedron_volume(clipping_tet) > 0);

		// Convert the subject tet to PolyClipper format.
		std::vector<Vertex3d<>> poly;
		{
			std::vector<Vector3d> positions(subject_tet.rows());
			for (int i = 0; i < subject_tet.rows(); i++)
				positions[i] = toPolyClipperVector(subject_tet.row(i));

			std::vector<std::vector<int>> neighbors = {{1, 3, 2}, {2, 3, 0}, {0, 3, 1}, {0, 1, 2}};

			initializePolyhedron(poly, positions, neighbors);
		}

		// Clip the subject tet with the clipping tet.
		{
			// Convert the clipping tet to planes.
			const Vector3d t0 = toPolyClipperVector(clipping_tet.row(0));
			const Vector3d t1 = toPolyClipperVector(clipping_tet.row(1));
			const Vector3d t2 = toPolyClipperVector(clipping_tet.row(2));
			const Vector3d t3 = toPolyClipperVector(clipping_tet.row(3));

			std::vector<Plane<internal::VectorAdapter<Vector3d>>> planes;
			planes.emplace_back(t0, (t1 - t0).cross(t2 - t0).unitVector());
			planes.emplace_back(t0, (t2 - t0).cross(t3 - t0).unitVector());
			planes.emplace_back(t0, (t3 - t0).cross(t1 - t0).unitVector());
			planes.emplace_back(t1, (t3 - t1).cross(t2 - t1).unitVector());

			clipPolyhedron(poly, planes);
		}

		// Convert the clipped polygon to Eigen format.
		const std::vector<std::vector<int>> T = splitIntoTetrahedra(poly);
		std::vector<Eigen::MatrixXd> result(T.size(), Eigen::MatrixXd(4, 3));
		for (int i = 0; i < T.size(); i++)
		{
			for (int j = 0; j < 4; j++)
				result[i].row(j) = fromPolyClipperVector(poly[T[i][j]].position);
			assert(tetrahedron_volume(result[i]) > 0);
		}

		return result;
	}

#ifdef POLYFEM_WITH_CLIPPER
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
#endif

	PolyClipper::Vector2d PolygonClipping::toPolyClipperVector(const Eigen::Vector2d &v)
	{
		return PolyClipper::Vector2d(v.x(), v.y());
	}

	Eigen::Vector2d PolygonClipping::fromPolyClipperVector(const PolyClipper::Vector2d &v)
	{
		return Eigen::Vector2d(v.x, v.y);
	}

	PolyClipper::Vector3d TetrahedronClipping::toPolyClipperVector(const Eigen::Vector3d &v)
	{
		return PolyClipper::Vector3d(v.x(), v.y(), v.z());
	}

	Eigen::Vector3d TetrahedronClipping::fromPolyClipperVector(const PolyClipper::Vector3d &v)
	{
		return Eigen::Vector3d(v.x, v.y, v.z);
	}
} // namespace polyfem::utils
