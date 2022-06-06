#include "PolygonQuadrature.hpp"
#include "TriQuadrature.hpp"

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
		} // namespace

		PolygonQuadrature::PolygonQuadrature()
		{
		}

		void PolygonQuadrature::get_quadrature(const Eigen::MatrixXd &poly, const int order, Quadrature &quadr)
		{
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

			Quadrature tri_quadr_pts;
			TriQuadrature tri_quadr;
			tri_quadr.get_quadrature(3, tri_quadr_pts);

			const long offset = tri_quadr_pts.weights.rows();
			quadr.points.resize(tris.rows() * offset, 2);
			quadr.weights.resize(tris.rows() * offset, 1);

			Eigen::MatrixXd trafod_pts;
			Eigen::Matrix<double, 3, 2> triangle;

			// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;

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

			assert(quadr.weights.minCoeff() >= 0);
		}
	} // namespace quadrature
} // namespace polyfem
