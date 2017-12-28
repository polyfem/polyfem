#include "ElementAssemblyValues.hpp"


namespace poly_fem
{
	void ElementAssemblyValues::finalize_global_element(const Eigen::MatrixXd &v)
	{
		val = v;

		has_parameterization = false;
		det.resize(v.rows(), 1);
		det.setConstant(1);

		for(std::size_t j = 0; j < basis_values.size(); ++j)
			basis_values[j].grad_t_m = basis_values[j].grad;
	}

	void ElementAssemblyValues::finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz)
	{
		val = v;

		det.resize(v.rows(), 1);

		for(std::size_t j = 0; j < basis_values.size(); ++j)
			basis_values[j].finalize();

		// std::cout<<"\n\ndx:\n"<<dx<<std::endl;
		// std::cout<<"\n\ndy:\n"<<dy<<std::endl;
		// std::cout<<"\n\ndz:\n"<<dz<<std::endl;

		Eigen::Matrix3d tmp;
		for(long i=0; i < v.rows(); ++i)
		{
			tmp.row(0) = dx.row(i);
			tmp.row(1) = dy.row(i);
			tmp.row(2) = dz.row(i);

			det(i) = (tmp.determinant());
			// std::cout<<det(i)<<std::endl;

			Eigen::MatrixXd jac_it = tmp.inverse().transpose();
			for(std::size_t j = 0; j < basis_values.size(); ++j)
				basis_values[j].grad_t_m.row(i) = basis_values[j].grad.row(i) * jac_it;
		}
	}

	void ElementAssemblyValues::finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy)
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

	void ElementAssemblyValues::compute_assembly_values(const bool is_volume, const std::vector< ElementBases > &bases, std::vector< ElementAssemblyValues > &values)
	{
		values.resize(bases.size());

		for(std::size_t i = 0; i < bases.size(); ++i)
		{
			const Quadrature &quadrature = bases[i].quadrature;
			const ElementBases &bs = bases[i];
			ElementAssemblyValues &vals = values[i];
			vals.basis_values.resize(bs.bases.size());
			vals.quadrature = quadrature;

			Eigen::MatrixXd mval = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

			Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
			Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
			Eigen::MatrixXd dzmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

			const int n_local_bases = int(bs.bases.size());
			for(int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b=bs.bases[j];
				AssemblyValues &val = vals.basis_values[j];

				val.global = b.global();


				b.basis(quadrature.points, val.val);
				b.grad(quadrature.points, val.grad);

				if(!bs.has_parameterization) continue;

				for(std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					for (long k = 0; k < val.val.rows(); ++k){
						mval.row(k) += val.val(k,0)    * b.global()[ii].node * b.global()[ii].val;

						dxmv.row(k) += val.grad(k,0) * b.global()[ii].node  * b.global()[ii].val;
						dymv.row(k) += val.grad(k,1) * b.global()[ii].node  * b.global()[ii].val;
						if(is_volume)
							dzmv.row(k) += val.grad(k,2) * b.global()[ii].node  * b.global()[ii].val;
					}
				}
			}

			if(!bs.has_parameterization)
				vals.finalize_global_element(quadrature.points);
			else
			{
				if(is_volume)
					vals.finalize(mval, dxmv, dymv, dzmv);
				else
					vals.finalize(mval, dxmv, dymv);
			}
		}
	}
}
