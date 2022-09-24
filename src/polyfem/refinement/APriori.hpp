#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/io/OutData.hpp>

namespace polyfem::refinement
{
	class APriori
	{
	private:
		APriori() {}

	public:
		/// compute a priori prefinement in 3d, fills disc_orders
		/// @param[in] mesh3d mesh
		static void p_refine(const mesh::Mesh &mesh,
							 const double B,
							 const bool h1_formula,
							 const int base_p,
							 const int discr_order_max,
							 io::OutStatsData &stats,
							 Eigen::VectorXi &disc_orders);

	private:
		/// compute a priori prefinement in 2d, fills disc_orders
		/// @param[in] mesh2d mesh
		static void p_refine(const mesh::Mesh2D &mesh2d,
							 const double B,
							 const bool h1_formula,
							 const int base_p,
							 const int discr_order_max,
							 io::OutStatsData &stats,
							 Eigen::VectorXi &disc_orders);

		/// compute a priori prefinement in 3d, fills disc_orders
		/// @param[in] mesh3d mesh
		static void p_refine(const mesh::Mesh3D &mesh3d,
							 const double B,
							 const bool h1_formula,
							 const int base_p,
							 const int discr_order_max,
							 io::OutStatsData &stats,
							 Eigen::VectorXi &disc_orders);
	};
} // namespace polyfem::refinement
