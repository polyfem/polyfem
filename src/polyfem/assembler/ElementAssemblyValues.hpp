#pragma once

#include <polyfem/assembler/AssemblyValues.hpp>
#include <polyfem/basis/ElementBases.hpp>

#include <vector>

namespace polyfem
{
	namespace assembler
	{
		/// stores per element basis values at given quadrature points and geometric mapping
		class ElementAssemblyValues
		{
		public:
			// m = number of quadrature points

			// vector of basis values and gradients at quadrature points for this element
			// each element samples a single basis function at the m quadrature points
			std::vector<AssemblyValues> basis_values;
			// inverse transpose jacobian of geom mapping at quadrature points
			std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> jac_it;

			// quadrature rule to use (stores points and weights)
			quadrature::Quadrature quadrature;
			bool is_volume_;
			int element_id;

			// img of quadrature points through the geom mapping (global pos in the mesh)
			Eigen::MatrixXd val; // R^{m x dim}

			// det(∑∇φᵢ⋅Nᵢ) det fo the jacobian of geometric mapping (constant for P1)
			Eigen::VectorXd det; // R^{m x 1}

			// only poly elements have no parameterization
			bool has_parameterization = true;

			/// computes the per element values at the local (ref el) points (pts)
			/// sets basis_values, jac_it, val, and det members
			void compute(const int el_index, const bool is_volume, const Eigen::MatrixXd &pts, const basis::ElementBases &basis, const basis::ElementBases &gbasis);

			/// computes quadrature points for given element then calls above (overloaded) compute function
			void compute(const int el_index, const bool is_volume, const basis::ElementBases &basis, const basis::ElementBases &gbasis);
			
			/// check if the element is flipped
			bool is_geom_mapping_positive(const bool is_volume, const basis::ElementBases &gbasis) const;

			Eigen::VectorXd eval_deformed_jacobian_determinant(const Eigen::VectorXd &disp) const;

		private:
			const basis::ElementBases *basis_, *gbasis_;
			std::vector<AssemblyValues> g_basis_values_cache_;

			void finalize_global_element(const Eigen::MatrixXd &v);

			/// void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy);
			/// void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz);
			
			/// compute Jacobians
			void finalize2d(const basis::ElementBases &gbasis, const std::vector<AssemblyValues> &gbasis_values);
			void finalize3d(const basis::ElementBases &gbasis, const std::vector<AssemblyValues> &gbasis_values);

			bool is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz) const;
			bool is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy) const;
		};
	} // namespace assembler
} // namespace polyfem
