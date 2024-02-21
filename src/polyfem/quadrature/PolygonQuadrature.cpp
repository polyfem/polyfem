#include "PolygonQuadrature.hpp"
#include "TriQuadrature.hpp"

#include <igl/predicates/ear_clipping.h>
#include <igl/write_triangle_mesh.h>

#ifdef POLYFEM_WITH_TRIANGLE
#include <igl/triangle/triangulate.h>
#endif

#include <vector>
#include <cassert>
#include <iostream>

namespace polyfem
{
	namespace quadrature
	{
		namespace
		{
			template <class TriMat>
			double transform_pts(const TriMat &tri, const Eigen::MatrixXd &pts, Eigen::MatrixXd &trafoed)
			{
				Eigen::Matrix2d trafo;
				trafo.row(0) = tri.row(1) - tri.row(0);
				trafo.row(1) = tri.row(2) - tri.row(0);

				trafoed = pts * trafo;

				trafoed.col(0).array() += tri(0, 0);
				trafoed.col(1).array() += tri(0, 1);

				return trafo.determinant();
			}

			void assign_quadrature(const Quadrature &tri_quadr_pts, const Eigen::MatrixXi &tris, const Eigen::MatrixXd pts, Quadrature &quadr)
			{
				Eigen::MatrixXd trafod_pts;
				Eigen::Matrix<double, 3, 2> triangle;

				const long offset = tri_quadr_pts.weights.rows();
				quadr.points.resize(tris.rows() * offset, 2);
				quadr.weights.resize(tris.rows() * offset, 1);

				for (long i = 0; i < tris.rows(); ++i)
				{
					const auto &indices = tris.row(i);

					triangle.row(0) = pts.row(indices(0));
					triangle.row(1) = pts.row(indices(1));
					triangle.row(2) = pts.row(indices(2));

					const double det = transform_pts(triangle, tri_quadr_pts.points, trafod_pts);
					quadr.points.block(i * offset, 0, trafod_pts.rows(), trafod_pts.cols()) = trafod_pts;
					quadr.weights.block(i * offset, 0, tri_quadr_pts.weights.rows(), tri_quadr_pts.weights.cols()) = tri_quadr_pts.weights * det;
				}
			}
		} // namespace

		PolygonQuadrature::PolygonQuadrature()
		{
		}

		void PolygonQuadrature::get_quadrature(const Eigen::MatrixXd &poly, const int order, Quadrature &quadr)
		{
			Quadrature tri_quadr_pts;
			TriQuadrature tri_quadr;
			tri_quadr.get_quadrature(order, tri_quadr_pts);

#ifdef POLYFEM_WITH_TRIANGLE
			Eigen::MatrixXi E(poly.rows(), 2);
			const Eigen::MatrixXd H(0, 2);
			const std::string flags = "Qzqa0.01";

			for (int i = 0; i < poly.rows(); ++i)
				E.row(i) << i, (i + 1) % poly.rows();

			Eigen::MatrixXi tris;
			Eigen::MatrixXd pts;

			igl::triangle::triangulate(poly, E, H, flags, pts, tris);
			assign_quadrature(tri_quadr_pts, tris, pts, quadr);

			Eigen::MatrixXd asd(pts.rows(), 3);
			asd.col(0) = pts.col(0);
			asd.col(1) = pts.col(1);
			asd.col(2).setZero();
			igl::write_triangle_mesh("quad.obj", asd, tris);

#else
			const int n_vertices = poly.rows();
			double area = 0;
			Eigen::Matrix2d tmp;

			Eigen::MatrixXi tris(n_vertices, 3);
			Eigen::MatrixXd pts(n_vertices + 1, 2);
			pts.row(n_vertices) = poly.colwise().mean();

			for (int e = 0; e < n_vertices; ++e)
			{
				const int ep = (e + 1) % n_vertices;
				tris.row(e) << e, ep, n_vertices;
				pts.row(e) = poly.row(e);

				tmp.row(0) = poly.row(e);
				tmp.row(1) = poly.row(ep);

				area += tmp.determinant();
			}

			area = fabs(area);

			assign_quadrature(tri_quadr_pts, tris, pts, quadr);

			// polygon is concave
			if (quadr.weights.minCoeff() < 0)
			{
				const Eigen::MatrixXi rt = Eigen::MatrixXi::Zero(poly.rows(), 1);
				tris.resize(0, 0);
				Eigen::VectorXi I;
				igl::predicates::ear_clipping(poly, rt, tris, I);

				assign_quadrature(tri_quadr_pts, tris, poly, quadr);
			}
#endif

			assert(quadr.weights.minCoeff() >= 0);
		}
	} // namespace quadrature
} // namespace polyfem
