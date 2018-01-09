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

			// for(int k=0; k < 3;++k){
			// 	for(int l=0; l < 3;++l)
			// 	{
			// 		tmp(k,l) = fabs(tmp(k,l))<1e-10 ? 0 : tmp(k,l);
			// 	}
			// }

			det(i) = tmp.determinant();
			// std::cout<<tmp<<std::endl;
			// assert(det(i)>0);

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

			det(i) = tmp.determinant();
			assert(det(i)>0);
			// std::cout<<det(i)<<std::endl;

				// std::cout<<"tmp.inverse().transpose() "<<tmp.inverse().transpose()<<std::endl;
			Eigen::MatrixXd jac_it = tmp.inverse().transpose();
			for(std::size_t j = 0; j < basis_values.size(); ++j)
				basis_values[j].grad_t_m.row(i) = basis_values[j].grad.row(i) * jac_it;
		}
	}

	void ElementAssemblyValues::compute(const bool is_volume, const ElementBases &basis)
	{
		quadrature = basis.quadrature;

		basis_values.resize(basis.bases.size());
		quadrature = quadrature;

		Eigen::MatrixXd mval = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

		Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
		Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
		Eigen::MatrixXd dzmv;

		if(is_volume)
			dzmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

		const int n_local_bases = int(basis.bases.size());
		for(int j = 0; j < n_local_bases; ++j)
		{
			const Basis &b=basis.bases[j];
			AssemblyValues &ass_val = basis_values[j];

			ass_val.global = b.global();


			b.basis(quadrature.points, ass_val.val);
			assert(ass_val.val.cols()==1);

			b.grad(quadrature.points, ass_val.grad);
			assert(ass_val.grad.cols() == quadrature.points.cols());

			if(!basis.has_parameterization) continue;

			for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			{
				for (long k = 0; k < ass_val.val.rows(); ++k)
				{
					mval.row(k) += ass_val.val(k)    * b.global()[ii].node * b.global()[ii].val;

					dxmv.row(k) += ass_val.grad(k,0) * b.global()[ii].node  * b.global()[ii].val;
					dymv.row(k) += ass_val.grad(k,1) * b.global()[ii].node  * b.global()[ii].val;
					if(is_volume)
						dzmv.row(k) += ass_val.grad(k,2) * b.global()[ii].node  * b.global()[ii].val;
				}
			}
		}

		if(!basis.has_parameterization)
			finalize_global_element(quadrature.points);
		else
		{
			if(is_volume)
				finalize(mval, dxmv, dymv, dzmv);
			else
				finalize(mval, dxmv, dymv);
		}
	}

	void ElementAssemblyValues::compute_assembly_values(const bool is_volume, const std::vector< ElementBases > &bases, std::vector< ElementAssemblyValues > &values)
	{
		values.resize(bases.size());

		for(std::size_t i = 0; i < bases.size(); ++i)
		{
			if (!bases[i].bases.empty() && bases[i].bases.front().is_defined()) {
				values[i].compute(is_volume, bases[i]);
			}
		}
	}
}
