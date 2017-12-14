#ifndef ELEMENT_ASSEMBLY_VALUES
#define ELEMENT_ASSEMBLY_VALUES

#include "AssemblyValues.hpp"

#include <vector>
#include <iostream>

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

		void finalize_global_element(const Eigen::MatrixXd &v)
		{
			val = v;

			has_parameterization = false;
			det.resize(v.rows(), 1);
			det.setConstant(-1);

			for(std::size_t j = 0; j < basis_values.size(); ++j)
				basis_values[j].grad_t_m = basis_values[j].grad;
		}

		void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz)
		{
			val = v;

			det.resize(v.rows(), 1);

			for(std::size_t j = 0; j < basis_values.size(); ++j)
				basis_values[j].finalize();

			Eigen::Matrix3d tmp;
			for(long i=0; i < v.rows(); ++i)
			{
				tmp.row(0) = dx.row(i);
				tmp.row(1) = dy.row(i);
				tmp.row(2) = dz.row(i);

				det(i) = (tmp.determinant());

				Eigen::MatrixXd jac_it = tmp.inverse().transpose();
				for(std::size_t j = 0; j < basis_values.size(); ++j)
					basis_values[j].grad_t_m.row(i) = basis_values[j].grad.row(i) * jac_it;
			}
		}

		void finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy)
		{
			val = v;

			det.resize(v.rows(), 1);

			for(std::size_t j = 0; j < basis_values.size(); ++j)
				basis_values[j].finalize();

			Eigen::Matrix2d tmp;
			for(long i = 0; i < v.rows(); ++i)
			{
				tmp.row(0) = dx.row(i);
				tmp.row(1) = dy.row(i);

				det(i) = (tmp.determinant());

				// std::cout<<"tmp.inverse().transpose() "<<tmp.inverse().transpose()<<std::endl;
				Eigen::MatrixXd jac_it = tmp.inverse().transpose();
				for(std::size_t j = 0; j < basis_values.size(); ++j)
					basis_values[j].grad_t_m.row(i) = basis_values[j].grad.row(i) * jac_it;
			}
		}
	};
}

#endif //ELEMENT_ASSEMBLY_VALUES
