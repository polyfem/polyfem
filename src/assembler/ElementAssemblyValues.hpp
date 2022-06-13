#pragma once

#include <polyfem/AssemblyValues.hpp>
#include <polyfem/ElementBases.hpp>

#include <vector>

namespace polyfem
{
	//stores per elment bases evaluation
	class ElementAssemblyValues
	{
	public:
		//per basis values
		std::vector<AssemblyValues> basis_values;
		//inverse transpose jacobian of geom mapping
		std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> jac_it;

		Quadrature quadrature;
		int element_id;

		// img of quadrature points through the geom mapping (global pos in the mesh)
		Eigen::MatrixXd val; // R^{m x dim}

		// det(∑∇φi.Ni) det fo the jacobian of geometric mapping (constant for P1)
		Eigen::VectorXd det; // R^{m x 1}

		//only poly elements have no parameterization
		bool has_parameterization = true;

		//computes the per element values at the quadrature points
		void compute(const int el_index, const bool is_volume, const ElementBases &basis, const ElementBases &gbasis);
		//computes the per element values at the local (ref el) points (pts)
		void compute(const int el_index, const bool is_volume, const Eigen::MatrixXd &pts, const ElementBases &basis, const ElementBases &gbasis);
		//check if the element is flipped
		bool is_geom_mapping_positive(const bool is_volume, const ElementBases &gbasis) const;
		bool is_geom_mapping_degenerate(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz) const;
		bool is_geom_mapping_degenerate(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy) const;

	private:
		std::vector<AssemblyValues> g_basis_values_cache_;

		void finalize_global_element(const Eigen::MatrixXd &v);

		// void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy);
		// void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz);
		void finalize2d(const ElementBases &gbasis, const std::vector<AssemblyValues> &gbasis_values);
		void finalize3d(const ElementBases &gbasis, const std::vector<AssemblyValues> &gbasis_values);

		bool is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz) const;
		bool is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy) const;
	};
} // namespace polyfem
