#ifndef ELEMENT_ASSEMBLY_VALUES
#define ELEMENT_ASSEMBLY_VALUES

#include <polyfem/AssemblyValues.hpp>
#include <polyfem/ElementBases.hpp>

#include <vector>

namespace poly_fem
{
	class ElementAssemblyValues
	{
	public:
		std::vector<AssemblyValues> basis_values;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> jac_it;

		Quadrature quadrature;

		// img of quadrature points through the geom mapping (global pos in the mesh)
		Eigen::MatrixXd val; // R^{m x dim}

		// det(∑∇φi.Ni) det fo the jacobian of geometric mapping (constant for P1)
		Eigen::VectorXd det; // R^{m x 1}

		bool has_parameterization = true;

		void compute(const int el_index, const bool is_volume, const ElementBases &basis, const ElementBases &gbasis);
		void compute(const int el_index, const bool is_volume, const Eigen::MatrixXd &pts, const ElementBases &basis, const ElementBases &gbasis);
		bool is_geom_mapping_positive(const bool is_volume, const ElementBases &gbasis) const;

	private:
		void finalize_global_element(const Eigen::MatrixXd &v);

		void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy);
		void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz);

		bool is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz) const;
		bool is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy) const;
	};
}

#endif //ELEMENT_ASSEMBLY_VALUES
