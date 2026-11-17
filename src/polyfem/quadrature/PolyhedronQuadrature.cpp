////////////////////////////////////////////////////////////////////////////////
#include "PolyhedronQuadrature.hpp"
#include "TetQuadrature.hpp"
#include <vector>
#include <cassert>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem
{
	////////////////////////////////////////////////////////////////////////////////

	void tertrahedralize_star_shaped_surface(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
											 const Eigen::RowVector3d &kernel, Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, Eigen::MatrixXi &OT)
	{
		assert(V.cols() == 3);
		OV.resize(V.rows() + 1, V.cols());
		OV.topRows(V.rows()) = V;
		OV.bottomRows(1) = kernel;
		OF = F;
		OT.resize(OF.rows(), 4);
		OT.col(0).setConstant(V.rows());
		OT.rightCols(3) = F;
	}

	namespace quadrature
	{
		namespace
		{

			template <class TriMat>
			double transform_pts(const TriMat &tri, const Eigen::MatrixXd &pts, Eigen::MatrixXd &transformed)
			{
				Eigen::Matrix3d matrix;
				matrix.row(0) = tri.row(1) - tri.row(0);
				matrix.row(1) = tri.row(2) - tri.row(0);
				matrix.row(2) = tri.row(3) - tri.row(0);

				transformed = pts * matrix;

				transformed.col(0).array() += tri(0, 0);
				transformed.col(1).array() += tri(0, 1);
				transformed.col(2).array() += tri(0, 2);

				return matrix.determinant();
			}

		} // anonymous namespace

		////////////////////////////////////////////////////////////////////////////////

		void PolyhedronQuadrature::get_quadrature(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
												  const Eigen::RowVector3d &kernel, const int order, Quadrature &quadr)
		{
			std::string flags = "Qpq2.0";
			Eigen::VectorXi J;
			Eigen::MatrixXd VV, OV, TV;
			Eigen::MatrixXi OF, TF, tets;

			Quadrature tet_quadr_pts;
			TetQuadrature tet_quadr;
			tet_quadr.get_quadrature(4, tet_quadr_pts);
			// assert(tet_quadr_pts.weights.minCoeff() >= 0);

			tertrahedralize_star_shaped_surface(V, F, kernel, TV, TF, tets);

			const long offset = tet_quadr_pts.weights.rows();
			quadr.points.resize(tets.rows() * offset, 3);
			quadr.weights.resize(tets.rows() * offset, 1);

			Eigen::MatrixXd transformed_points;

			for (long i = 0; i < tets.rows(); ++i)
			{
				Eigen::Matrix<double, 4, 3> tetra;
				const auto &indices = tets.row(i);
				tetra.row(0) = TV.row(indices(0));
				tetra.row(1) = TV.row(indices(1));
				tetra.row(2) = TV.row(indices(2));
				tetra.row(3) = TV.row(indices(3));

				// viewer.data.add_edges(triangle.row(0), triangle.row(1), Eigen::Vector3d(1,0,0).transpose());
				// viewer.data.add_edges(triangle.row(0), triangle.row(2), Eigen::Vector3d(1,0,0).transpose());
				// viewer.data.add_edges(triangle.row(2), triangle.row(1), Eigen::Vector3d(1,0,0).transpose());

				const double det = transform_pts(tetra, tet_quadr_pts.points, transformed_points);
				assert(det > 0);

				quadr.points.block(i * offset, 0, transformed_points.rows(), transformed_points.cols()) = transformed_points;
				quadr.weights.block(i * offset, 0, tet_quadr_pts.weights.rows(), tet_quadr_pts.weights.cols()) = tet_quadr_pts.weights * det;
			}

			// assert(quadr.weights.minCoeff() >= 0);
		}
	} // namespace quadrature
} // namespace polyfem
