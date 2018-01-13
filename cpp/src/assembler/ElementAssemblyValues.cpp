#include "ElementAssemblyValues.hpp"
#include <igl/Timer.h>

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

	bool ElementAssemblyValues::is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz) const
	{
		Eigen::Matrix3d tmp;
		for(long i=0; i < dx.rows(); ++i)
		{
			tmp.row(0) = dx.row(i);
			tmp.row(1) = dy.row(i);
			tmp.row(2) = dz.row(i);

			if(tmp.determinant() <= 0)
				return false;
		}

		return true;
	}

	bool ElementAssemblyValues::is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy) const
	{
		Eigen::Matrix2d tmp;
		for(long i = 0; i < dx.rows(); ++i)
		{
			tmp.row(0) = dx.row(i);
			tmp.row(1) = dy.row(i);

			if(tmp.determinant() <= 0)
				return false;
		}

		return true;
	}

	void ElementAssemblyValues::finalize(const int el_index, const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz)
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

	void ElementAssemblyValues::compute(const int el_index, const bool is_volume, const ElementBases &basis)
	{
		basis.compute_quadrature(quadrature);

		bool poly = (quadrature.weights.size() > 1000);

		basis_values.resize(basis.bases.size());

		Eigen::MatrixXd mval = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

		Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
		Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
		Eigen::MatrixXd dzmv;

		if(is_volume)
			dzmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

		double t = 0;
		const int n_local_bases = int(basis.bases.size());
		for(int j = 0; j < n_local_bases; ++j)
		{
			const Basis &b=basis.bases[j];
			AssemblyValues &ass_val = basis_values[j];

			ass_val.global = b.global();

			igl::Timer timer0;
			timer0.start();

			b.basis(quadrature.points, ass_val.val);
			assert(ass_val.val.cols()==1);

			b.grad(quadrature.points, ass_val.grad);
			assert(ass_val.grad.cols() == quadrature.points.cols());

			timer0.stop();
			t += timer0.getElapsedTime();

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
		if (poly) { std::cout << "-- eval quadr points: " << t << std::endl; }

		if(!basis.has_parameterization) {
			finalize_global_element(quadrature.points);
		}
		else
		{
			if(is_volume)
				finalize(el_index, mval, dxmv, dymv, dzmv);
			else
				finalize(mval, dxmv, dymv);
		}

	}

	bool ElementAssemblyValues::is_geom_mapping_positive(const bool is_volume, const ElementBases &basis) const
	{
		if(!basis.has_parameterization)
			return true;

		Quadrature quad;
		basis.compute_quadrature(quad);


		Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());
		Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());
		Eigen::MatrixXd dzmv;
		Eigen::MatrixXd grad;

		if(is_volume)
			dzmv = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());

		double t = 0;
		const int n_local_bases = int(basis.bases.size());
		for(int j = 0; j < n_local_bases; ++j)
		{
			const Basis &b=basis.bases[j];

			b.grad(quad.points, grad);
			assert(grad.cols() == quad.points.cols());

			for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			{
				for (long k = 0; k < grad.rows(); ++k)
				{
					dxmv.row(k) += grad(k,0) * b.global()[ii].node  * b.global()[ii].val;
					dymv.row(k) += grad(k,1) * b.global()[ii].node  * b.global()[ii].val;
					if(is_volume)
						dzmv.row(k) += grad(k,2) * b.global()[ii].node  * b.global()[ii].val;
				}
			}
		}

		return is_volume ? is_geom_mapping_positive(dxmv, dymv, dzmv) : is_geom_mapping_positive(dxmv, dxmv);
	}

	// void ElementAssemblyValues::compute_assembly_values(const bool is_volume, const std::vector< ElementBases > &bases, std::vector< ElementAssemblyValues > &values)
	// {
	// 	values.resize(bases.size());

	// 	for(std::size_t i = 0; i < bases.size(); ++i)
	// 	{
	// 		if (!bases[i].bases.empty() && bases[i].bases.front().is_defined()) {
	// 			values[i].compute(i, is_volume, bases[i]);
	// 		}
	// 	}
	// }
}
