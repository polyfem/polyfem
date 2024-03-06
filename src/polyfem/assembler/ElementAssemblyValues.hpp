#pragma once

#include <polyfem/assembler/AssemblyValues.hpp>
#include <polyfem/basis/ElementBases.hpp>

#include <vector>

namespace polyfem
{
	namespace assembler
	{
		/// @brief Per element basis values at given quadrature points and geometric mapping.
		/// @note \f$m\f$ = number of quadrature points; \f$d\f$ = dimension
		class ElementAssemblyValues
		{
		public:
			/// @brief Vector of basis values and gradients at quadrature points for this element.
			/// Each element samples a single basis function at the m quadrature points.
			std::vector<AssemblyValues> basis_values;
			/// @brief The inverse transpose of the Jacobian of the geometry mapping at quadrature points.
			/// Stands for <b>Jac</b>obian <b>i</b>nverse <b>t</b>ranspose.
			std::vector<MatrixNd> jac_it;

			/// @brief Quadrature rule to use (stores points and weights).
			quadrature::Quadrature quadrature;
			/// @brief Index of the element in the mesh.
			int element_id;

			/// @brief Image of quadrature points through the geometry mapping (global position in the mesh).
			/// \f$\mathbb{R}^{m x d}\f$
			Eigen::MatrixXd val;

			/// @brief Determinant of the Jacobian of geometric mapping.
			/// \f$\det(\sum \nabla \phi_i \cdot N_i)\f$ (constant for P1)
			Eigen::VectorXd det; // R^{m x 1}

			/// @brief Is the element defined by a reference domain?
			/// Only polygonal elements have no parameterization.
			bool has_parameterization = true;

			/// @brief Compute the per element values at the local points.
			/// @param el_index Index of the element in the mesh.
			/// @param is_volume Is the element volumetric?
			/// @param pts Local points to evaluate the basis functions at.
			/// @param basis Basis functions for the element.
			/// @param gbasis Geometric basis functions for the element.
			/// @note This function sets basis_values, jac_it, val, and det members.
			void compute(
				const int el_index,
				const bool is_volume,
				const Eigen::MatrixXd &pts,
				const basis::ElementBases &basis,
				const basis::ElementBases &gbasis);

			/// @brief Compute quadrature points for given element then calls compute().
			/// @param el_index Index of the element in the mesh.
			/// @param is_volume Is the element volumetric?
			/// @param basis Basis functions for the element.
			/// @param gbasis Geometric basis functions for the element.
			void compute(
				const int el_index,
				const bool is_volume,
				const basis::ElementBases &basis,
				const basis::ElementBases &gbasis);

			/// @brief Check if the element is flipped.
			bool is_geom_mapping_positive(const bool is_volume, const basis::ElementBases &gbasis) const;

		private:
			/// @brief Cache of basis values and gradients at quadrature points for the geometric basis.
			std::vector<AssemblyValues> g_basis_values_cache_;

			/// @brief Compute Jacobians.
			/// @param gbasis Geometric basis functions for the element.
			/// @param gbasis_values Basis values and gradients at quadrature points for the geometric basis.
			/// @param is_volume Is the element volumetric?
			void finalize(
				const basis::ElementBases &gbasis,
				const std::vector<AssemblyValues> &gbasis_values,
				const bool is_volume);

			/// @brief Compute Jacobians.
			/// @param v Global positions of the quadrature points.
			void finalize_global_element(const Eigen::MatrixXd &v);

			/// @brief Check if the element is flipped.
			/// @param dx Displacement in x.
			/// @param dy Displacement in y.
			/// @param dz (optional) Displacement in z.
			/// @return True if the element is not flipped.
			bool is_geom_mapping_positive(
				const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz) const;
		};
	} // namespace assembler
} // namespace polyfem
