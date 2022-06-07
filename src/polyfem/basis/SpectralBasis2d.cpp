#include "SpectralBasis2d.hpp"

#include "QuadraticBSpline2d.hpp"
#include <polyfem/quadrature/QuadQuadrature.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/basis/FEBasis2d.hpp>
#include <polyfem/utils/Types.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Sparse>

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <map>

namespace polyfem
{
	namespace basis
	{
		using namespace Eigen;

		namespace
		{
			void basis(const Eigen::MatrixXd &uv, const int n, const int m, Eigen::MatrixXd &result)
			{
				const int n_pts = int(uv.rows());
				assert(uv.cols() == 2);

				result.resize(n_pts, 1);

				for (int i = 0; i < n_pts; ++i)
					result(i) = sin(n * M_PI * uv(i, 0)) * sin(m * M_PI * uv(i, 1));
			}

			void derivative(const Eigen::MatrixXd &uv, const int n, const int m, Eigen::MatrixXd &result)
			{
				const int n_pts = int(uv.rows());
				assert(uv.cols() == 2);

				result.resize(n_pts, 2);

				for (int i = 0; i < n_pts; ++i)
				{
					const double u = uv(i, 0);
					const double v = uv(i, 1);

					result(i, 0) = cos(n * M_PI * uv(i, 0)) * sin(m * M_PI * uv(i, 1));
					result(i, 1) = sin(n * M_PI * uv(i, 0)) * cos(m * M_PI * uv(i, 1));
				}
			}
		} // namespace

		int SpectralBasis2d::build_bases(
			const mesh::Mesh2D &mesh,
			const int quadrature_order,
			const int order,
			std::vector<ElementBases> &bases,
			std::vector<ElementBases> &gbases,
			std::vector<mesh::LocalBoundary> &local_boundary)
		{
			bases.resize(1);
			ElementBases &b = bases.front();
			b.has_parameterization = false;

			const int n_bases = order * order;

			b.bases.resize(n_bases);
			b.set_quadrature([quadrature_order](Quadrature &quad) {
				QuadQuadrature quad_quadrature;
				quad_quadrature.get_quadrature(quadrature_order, quad);
			});

			for (int i = 0; i < order; ++i)
			{
				for (int j = 0; j < order; ++j)
				{
					const int global_index = order * i + j;
					assert(global_index < n_bases);

					b.bases[global_index].init(-3, global_index, j, Eigen::MatrixXd::Zero(1, 2));
					b.bases[global_index].set_basis([i, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { basis(uv, i, j, val); });
					b.bases[global_index].set_grad([i, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { derivative(uv, i, j, val); });
				}
			}

			// gbases.resize(1);
			// ElementBases &gb = bases.front();
			// gb.bases.resize(n_bases);
			// gb.set_quadrature([quadrature_order](Quadrature &quad){
			//     QuadQuadrature quad_quadrature;
			//     quad_quadrature.get_quadrature(quadrature_order, quad);
			// });
			// b.has_parameterization = false;

			// for (int j = 0; j < n_bases; ++j) {
			//     const int global_index = j;

			//     gb.bases[j].init(global_index, j, Eigen::Vector2d(0,0));
			//     gb.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { basis(uv, j, val); });
			//     gb.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { derivative(uv, j, val); });
			// }

			return n_bases;
		}
	} // namespace basis
} // namespace polyfem
