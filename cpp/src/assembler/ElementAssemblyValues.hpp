#ifndef ELEMENT_ASSEMBLY_VALUES
#define ELEMENT_ASSEMBLY_VALUES

#include "AssemblyValues.hpp"
#include "ElementBases.hpp"

#include <vector>

namespace poly_fem
{
	class ElementAssemblyValues
	{
	public:
		std::vector<AssemblyValues> basis_values;
		Quadrature quadrature;

		Eigen::MatrixXd val;
		Eigen::MatrixXd det;

		bool has_parameterization = true;

		void finalize_global_element(const Eigen::MatrixXd &v);

		void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy);
		void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz);

		static void compute_assembly_values(const bool is_volume, const std::vector< ElementBases > &bases, std::vector< ElementAssemblyValues > &values);
	};
}

#endif //ELEMENT_ASSEMBLY_VALUES
