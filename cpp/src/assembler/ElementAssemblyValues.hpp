#ifndef ELEMENT_ASSEMBLY_VALUES
#define ELEMENT_ASSEMBLY_VALUES

#include "AssemblyValues.hpp"

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

		void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz)
		{
			val = v;

			det.resize(v.rows(), 1);

			Eigen::Matrix3d tmp;
			for(long i=0; i < v.rows(); ++i)
			{
				tmp.row(0) = dx.row(i);
				tmp.row(1) = dy.row(i);
				tmp.row(2) = dz.row(i);

				det(i) = (tmp.determinant());

				for(std::size_t j = 0; j < basis_values.size(); ++j)
					basis_values[j].finalize(tmp.inverse().transpose());
			}
		}

		void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy)
		{
			val = v;

			det.resize(v.rows(), 1);

			Eigen::Matrix2d tmp;
			for(long i=0; i < v.rows(); ++i)
			{
				tmp.row(0) = dx.row(i);
				tmp.row(1) = dy.row(i);

				det(i) = (tmp.determinant());

				for(std::size_t j = 0; j < basis_values.size(); ++j){
					basis_values[j].finalize(tmp.inverse().transpose());
				}
			}
		}
	};
}

#endif //ELEMENT_ASSEMBLY_VALUES
