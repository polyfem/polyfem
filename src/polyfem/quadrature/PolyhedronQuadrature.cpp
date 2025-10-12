////////////////////////////////////////////////////////////////////////////////
#include "PolyhedronQuadrature.hpp"
#include "TetQuadrature.hpp"
#include <polyfem/mesh/MeshUtils.hpp>
#include <geogram/mesh/mesh_io.h>
#include <igl/writeMESH.h>
#ifdef POLYFEM_WITH_MMG
#include <boost/filesystem.hpp>
#endif
#include <vector>
#include <cassert>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem
{
	namespace quadrature
	{
		namespace
		{

#ifdef POLYFEM_WITH_MMG

			// try to remesh volume if polyhedron creates more than this # of quadrature points
			const int max_num_quadrature_points = 2048;

			bool mmg_remesh_volume(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXi &T,
								   Eigen::MatrixXd &TV, Eigen::MatrixXi &TF, Eigen::MatrixXi &TT)
			{
				using namespace boost;

				double scaling = (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
				Eigen::RowVector3d translation = V.colwise().minCoeff();

				auto tmp_dir = filesystem::temp_directory_path();
				auto base_path = tmp_dir / filesystem::unique_path("polyfem_%%%%-%%%%-%%%%-%%%%");
				auto f_input = base_path;
				f_input += "_in.mesh";
				auto f_output = base_path;
				f_output += "_out.mesh";
				auto f_sol = base_path;
				f_sol += "_out.sol";

				TV = (V.rowwise() - translation) / scaling;
				igl::writeMESH(f_input.string(), TV, T, F);

				std::string app(POLYFEM_MMG_PATH);
				std::string cmd = app + " -ar 20 -hausd 0.01 -v 0 -in " + f_input.string() + " -out " + f_output.string();
#ifndef WIN32
				cmd += " &> /dev/null";
#endif
				logger().trace("Running command:\n {}", cmd);
				if (::system(cmd.c_str()) == 0)
				{
					GEO::Mesh M;
					GEO::mesh_load(f_output.string(), M);
					from_geogram_mesh(M, TV, TF, TT);
					TV = (scaling * TV).rowwise() + translation;

					filesystem::remove(f_input);
					filesystem::remove(f_output);
					filesystem::remove(f_sol);
					return true;
				}
				else
				{
					filesystem::remove(f_input);
					return false;
				}
			}

#endif

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

			double scaling = (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
			Eigen::RowVector3d translation = V.colwise().minCoeff();

			polyfem::mesh::tertrahedralize_star_shaped_surface(V, F, kernel, TV, TF, tets);

#ifdef POLYFEM_WITH_MMG
			if (tet_quadr_pts.weights.size() * tets.rows() > max_num_quadrature_points)
			{
				Eigen::MatrixXd V0;
				Eigen::MatrixXi F0, T0;
				bool res = mmg_remesh_volume(TV, TF, tets, V0, F0, T0);

				if (res && T0.rows() < tets.rows())
				{
					TV = V0;
					TF = F0;
					tets = T0;
				}
			}
#endif

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
