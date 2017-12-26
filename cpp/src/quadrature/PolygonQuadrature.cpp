#include "PolygonQuadrature.hpp"
#include "TriQuadrature.hpp"

#include "UIState.hpp"

#include <igl/triangle/triangulate.h>

#include <vector>
#include <cassert>
#include <iostream>

namespace poly_fem
{
    namespace
    {
        template<class TriMat>
        double transform_pts(const TriMat &tri, const Eigen::MatrixXd &pts, Eigen::MatrixXd &trafoed)
        {
            Eigen::Matrix2d trafo;
            trafo.row(0) = tri.row(1) - tri.row(0);
            trafo.row(1) = tri.row(2) - tri.row(0);

            trafoed = pts * trafo;

            trafoed.col(0).array() += tri(0,0);
            trafoed.col(1).array() += tri(0,1);

            return trafo.determinant();
        }
    }

    PolygonQuadrature::PolygonQuadrature()
    { }

    void PolygonQuadrature::get_quadrature(const Eigen::MatrixXd &poly, const int order, Quadrature &quad)
    {
        double area = 0;
        Eigen::MatrixXi E(poly.rows(),2);
        Eigen::Matrix2d tmp;
        for(int e = 0; e < int(poly.rows()); ++e)
        {
            const int ep = (e+1) % poly.rows();

            E(e, 1) = e;
            E(e, 0) = ep;

            tmp.row(0) = poly.row(e);
            tmp.row(1) = poly.row(ep);

            area += tmp.determinant();
        }

        area = fabs(area);

        Eigen::MatrixXi tris;
        Eigen::MatrixXd pts;
        std::stringstream ss;
        ss.precision(100);
        ss.setf(std::ios::fixed, std::ios::floatfield);
        ss<<"Qpa"<<0.00001/area;

        Eigen::MatrixXd poly_tmp(poly.rows()+1, 2);
        poly_tmp.block(0, 0, poly.rows(), 2) = poly;
        poly_tmp.block(poly.rows(), 0, 1, 2) = poly.colwise().mean(); //TODO replace with point in kernel

        igl::triangle::triangulate(poly_tmp, E, Eigen::MatrixXd(0,2), ss.str(), pts, tris);

        Quadrature tri_quad_pts;
        TriQuadrature tri_quad;
        tri_quad.get_quadrature(order, tri_quad_pts);

        const long offset = tri_quad_pts.weights.rows();
        quad.points.resize(tris.rows()*offset, 2);
        quad.weights.resize(tris.rows()*offset, 1);

        Eigen::MatrixXd trafod_pts;
        Eigen::Matrix<double, 3, 2> triangle;

        // igl::viewer::Viewer &viewer = UIState::ui_state().viewer;

        for(long i = 0; i < tris.rows(); ++i)
        {
            const auto &indices = tris.row(i);

            triangle.row(0) = pts.row(indices(0));
            triangle.row(1) = pts.row(indices(1));
            triangle.row(2) = pts.row(indices(2));

            // viewer.data.add_edges(triangle.row(0), triangle.row(1), Eigen::Vector3d(1,0,0).transpose());
            // viewer.data.add_edges(triangle.row(0), triangle.row(2), Eigen::Vector3d(1,0,0).transpose());
            // viewer.data.add_edges(triangle.row(2), triangle.row(1), Eigen::Vector3d(1,0,0).transpose());

            const double det = transform_pts(triangle, tri_quad_pts.points, trafod_pts);
            quad.points.block(i*offset, 0, trafod_pts.rows(), trafod_pts.cols()) = trafod_pts;
            quad.weights.block(i*offset, 0, tri_quad_pts.weights.rows(), tri_quad_pts.weights.cols()) = tri_quad_pts.weights * det;
        }
    }
}
