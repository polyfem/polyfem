#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/io/OutData.hpp>

namespace polyfem::refinement
{
	/// Class for a priori refinement, see 'Decoupling .. ' paper for details
	class APriori
	{
	private:
		APriori() {}

	public:
		/// compute a priori prefinement
		/// @param[in] mesh mesh
		/// @param[in] B The target deviation of the error on elements from perfect element error, for a priori geometry-dependent p-refinement, see 'Decoupling .. ' paper.
		/// @param[in] h1_formula lgacy code
		/// @param[in] base_p base element degree
		/// @param[in] discr_order_max maximum element degree
		/// @param[out] stats statistics to record angles, etc
		/// @param[out] disc_orders output per element order, assumes the array has correct size
		static void p_refine(const mesh::Mesh &mesh,
							 const double B,
							 const bool h1_formula,
							 const int base_p,
							 const int discr_order_max,
							 io::OutStatsData &stats,
							 Eigen::VectorXi &disc_orders);

	private:
		/// compute a priori prefinement in 2d
		/// @param[in] mesh2d mesh
		/// @param[in] B The target deviation of the error on elements from perfect element error, for a priori geometry-dependent p-refinement, see 'Decoupling .. ' paper.
		/// @param[in] h1_formula lgacy code
		/// @param[in] base_p base element degree
		/// @param[in] discr_order_max maximum element degree
		/// @param[out] stats statistics to record angles, etc
		/// @param[out] disc_orders output per element order, assumes the array has correct size
		static void p_refine(const mesh::Mesh2D &mesh2d,
							 const double B,
							 const bool h1_formula,
							 const int base_p,
							 const int discr_order_max,
							 io::OutStatsData &stats,
							 Eigen::VectorXi &disc_orders);

		/// compute a priori prefinement in 3d
		/// @param[in] mesh3d mesh
		/// @param[in] B The target deviation of the error on elements from perfect element error, for a priori geometry-dependent p-refinement, see 'Decoupling .. ' paper.
		/// @param[in] h1_formula lgacy code
		/// @param[in] base_p base element degree
		/// @param[in] discr_order_max maximum element degree
		/// @param[out] stats statistics to record angles, etc
		/// @param[out] disc_orders output per element order, assumes the array has correct size
		static void p_refine(const mesh::Mesh3D &mesh3d,
							 const double B,
							 const bool h1_formula,
							 const int base_p,
							 const int discr_order_max,
							 io::OutStatsData &stats,
							 Eigen::VectorXi &disc_orders);
	};
} // namespace polyfem::refinement
