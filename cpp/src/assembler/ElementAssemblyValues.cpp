#include "ElementAssemblyValues.hpp"
#include <igl/Timer.h>
#include <memory>

namespace poly_fem
{
	void ElementAssemblyValues::finalize_global_element(const Eigen::MatrixXd &v)
	{
		val = v;

		has_parameterization = false;
		det.resize(v.rows(), 1);

		jac_it.resize(v.rows());
		for(long i=0; i < v.rows(); ++i)
			jac_it[i] = Eigen::MatrixXd::Identity(v.cols(), v.cols());
		
		det.setConstant(1); // volume (det of the geometric mapping)
		for(std::size_t j = 0; j < basis_values.size(); ++j)
			basis_values[j].grad_t_m = basis_values[j].grad; // / scaling

	}

	bool ElementAssemblyValues::is_geom_mapping_positive(const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz) const
	{
		Eigen::Matrix3d tmp;
		for(long i=0; i < dx.rows(); ++i)
		{
			tmp.row(0) = dx.row(i);
			tmp.row(1) = dy.row(i);
			tmp.row(2) = dz.row(i);

			if(tmp.determinant() <= 0){
				// std::cout<<tmp.determinant()<<std::endl;
				return false;
			}
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

			if(tmp.determinant() <= 0){
				// std::cout<<tmp.determinant()<<std::endl;
				return false;
			}
		}

		return true;
	}

	void ElementAssemblyValues::finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz)
	{
		val = v;

		det.resize(v.rows(), 1);

		for(std::size_t j = 0; j < basis_values.size(); ++j)
			basis_values[j].finalize();

		Eigen::Matrix3d tmp;
		jac_it.resize(v.rows());
		for(long i=0; i < v.rows(); ++i)
		{
			tmp.row(0) = dx.row(i);
			tmp.row(1) = dy.row(i);
			tmp.row(2) = dz.row(i);

			det(i) = tmp.determinant();
			// std::cout<<det(i)<<std::endl;
			// assert(det(i)>0);


			jac_it[i] = tmp.inverse().transpose();
			for(std::size_t j = 0; j < basis_values.size(); ++j)
				basis_values[j].grad_t_m.row(i) = basis_values[j].grad.row(i) * jac_it[i];
		}
	}

	void ElementAssemblyValues::finalize(const Eigen::MatrixXd &v, const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy)
	{
		val = v;

		det.resize(v.rows(), 1);

		for(std::size_t j = 0; j < basis_values.size(); ++j)
			basis_values[j].finalize();

		Eigen::Matrix2d tmp;
		jac_it.resize(v.rows());
		for(long i = 0; i < v.rows(); ++i)
		{
			tmp.row(0) = dx.row(i);
			tmp.row(1) = dy.row(i);

			det(i) = tmp.determinant();
			// assert(det(i)>0);
			// std::cout<<det(i)<<std::endl;

			jac_it[i] = tmp.inverse().transpose();
			for(std::size_t j = 0; j < basis_values.size(); ++j)
				basis_values[j].grad_t_m.row(i) = basis_values[j].grad.row(i) * jac_it[i];
		}
	}


	void ElementAssemblyValues::compute(const int el_index, const bool is_volume, const ElementBases &basis, const ElementBases &gbasis)
	{
		basis.compute_quadrature(quadrature);
		compute(el_index, is_volume, quadrature.points, basis, gbasis);
	}

	void ElementAssemblyValues::compute(const int el_index, const bool is_volume, const Eigen::MatrixXd &pts, const ElementBases &basis, const ElementBases &gbasis)
	{
		// const bool poly = !gbasis.has_parameterization;

		basis_values.resize(basis.bases.size());

		Eigen::MatrixXd mval = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

		Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
		Eigen::MatrixXd dzmv = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

		igl::Timer timer0;
		timer0.start();

		auto values_and_grads = [&](const ElementBases &my_basis, auto v, auto dx, auto dy, auto dz)
		{
			my_basis.evaluate_bases(pts, *v);
			my_basis.evaluate_grads(pts, 0, *dx);
			my_basis.evaluate_grads(pts, 1, *dy);
			if(is_volume)
				my_basis.evaluate_grads(pts, 2, *dz);
		};

		auto vals = std::make_shared<Eigen::MatrixXd>();
		auto local_gradx = std::make_shared<Eigen::MatrixXd>();
		auto local_grady = std::make_shared<Eigen::MatrixXd>();
		auto local_gradz = std::make_shared<Eigen::MatrixXd>();
		values_and_grads(basis, vals, local_gradx, local_grady, local_gradz);

		auto gvals = std::make_shared<Eigen::MatrixXd>();
		auto local_g_gradx = std::make_shared<Eigen::MatrixXd>();
		auto local_g_grady = std::make_shared<Eigen::MatrixXd>();
		auto local_g_gradz = std::make_shared<Eigen::MatrixXd>();
		if (&basis == &gbasis) {
			gvals = vals;
			local_g_gradx = local_gradx;
			local_g_grady = local_grady;
			local_g_gradz = local_gradz;
		} else {
			values_and_grads(gbasis, gvals, local_g_gradx, local_g_grady, local_g_gradz);
		}

		timer0.stop();
		// const double t = timer0.getElapsedTime();
		// if (poly) { std::cout << "-- eval quadr points: " << t << std::endl; }


		const int n_local_bases = int(basis.bases.size());
		const int n_local_g_bases = int(gbasis.bases.size());

		for(int j = 0; j < n_local_bases; ++j)
		{
			AssemblyValues &ass_val = basis_values[j];

			ass_val.global = basis.bases[j].global();
			ass_val.val = vals->col(j);
			assert(ass_val.val.cols()==1);

			ass_val.grad.resize(pts.rows(), pts.cols());

			ass_val.grad.col(0) = local_gradx->col(j);
			ass_val.grad.col(1) = local_grady->col(j);
			if(is_volume)
				ass_val.grad.col(2) = local_gradz->col(j);

			assert(ass_val.grad.cols() == pts.cols());
		}

		if(!gbasis.has_parameterization) {
			// v = G(pts)
			finalize_global_element(pts);
			return;
		}

		for(int j = 0; j < n_local_g_bases; ++j)
		{
			const Basis &b=gbasis.bases[j];

			assert(gbasis.has_parameterization);

			for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			{
				for (long k = 0; k < gvals->rows(); ++k)
				{
					mval.row(k) += (*gvals)(k,j)    * b.global()[ii].node * b.global()[ii].val;

					dxmv.row(k) += (*local_g_gradx)(k,j) * b.global()[ii].node  * b.global()[ii].val;
					dymv.row(k) += (*local_g_grady)(k,j) * b.global()[ii].node  * b.global()[ii].val;
					if(is_volume)
						dzmv.row(k) += (*local_g_gradz)(k,j) * b.global()[ii].node  * b.global()[ii].val;
				}
			}
		}

		if(is_volume)
			finalize(mval, dxmv, dymv, dzmv);
		else
			finalize(mval, dxmv, dymv);

	}

	bool ElementAssemblyValues::is_geom_mapping_positive(const bool is_volume, const ElementBases &gbasis) const
	{
		if(!gbasis.has_parameterization)
			return true;

		const int n_local_bases = int(gbasis.bases.size());

		if(n_local_bases <= 0)
			return true;


		Quadrature quad;
		gbasis.compute_quadrature(quad);

		Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());
		Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());
		Eigen::MatrixXd dzmv;

		if(is_volume)
			dzmv = Eigen::MatrixXd::Zero(quad.points.rows(), quad.points.cols());

		Eigen::MatrixXd local_gradx, local_grady, local_gradz;
		gbasis.evaluate_grads(quad.points, 0, local_gradx);
		gbasis.evaluate_grads(quad.points, 1, local_grady);
		if(is_volume)
			gbasis.evaluate_grads(quad.points, 2, local_gradz);

		for(int j = 0; j < n_local_bases; ++j)
		{
			const Basis &b=gbasis.bases[j];

			// b.grad(quad.points, grad);
			// assert(grad.cols() == quad.points.cols());

			for(std::size_t ii = 0; ii < b.global().size(); ++ii)
			{
				for (long k = 0; k < local_gradx.rows(); ++k)
				{
					dxmv.row(k) += local_gradx(k,j) * b.global()[ii].node  * b.global()[ii].val;
					dymv.row(k) += local_grady(k,j) * b.global()[ii].node  * b.global()[ii].val;
					if(is_volume)
						dzmv.row(k) += local_gradz(k,j) * b.global()[ii].node  * b.global()[ii].val;
				}
			}
		}

		return is_volume ? is_geom_mapping_positive(dxmv, dymv, dzmv) : is_geom_mapping_positive(dxmv, dymv);
	}
}
