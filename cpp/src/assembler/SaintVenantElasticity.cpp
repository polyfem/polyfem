#include "SaintVenantElasticity.hpp"

#include "Basis.hpp"

#include <igl/Timer.h>


namespace poly_fem
{
	namespace
	{
		template<class Matrix>
		Matrix strain_from_disp_grad(const Matrix &disp_grad)
		{
			// Matrix mat =  (disp_grad + disp_grad.transpose());
			Matrix mat = (disp_grad.transpose()*disp_grad + disp_grad + disp_grad.transpose());

			for(int i = 0; i < mat.size(); ++i)
				mat(i) *= 0.5;

			return mat;
		}

		template<int dim>
		Eigen::Matrix<double, dim, dim> strain(const Eigen::MatrixXd &grad, const Eigen::MatrixXd &jac_it, int k, int coo)
		{
			Eigen::Matrix<double, dim, dim> jac;
			jac.setZero();
			jac.row(coo) = grad.row(k);
			jac = jac*jac_it;

			return strain_from_disp_grad(jac);
		}
	}



	SaintVenantElasticity::SaintVenantElasticity()
	{
		set_size(size_);
	}

	void SaintVenantElasticity::set_parameters(const json &params)
	{
		set_size(params["size"]);

		if(params["elasticity_tensor"].empty())
		{
			set_lambda_mu(params["lambda"], params["mu"]);
		}
		else
		{
			std::vector<double> entries = params["elasticity_tensor"];
			elasticity_tensor_.set_from_entries(entries);
		}
	}

	void SaintVenantElasticity::set_size(const int size)
	{
		elasticity_tensor_.resize(size);
		size_ = size;
	}

	template <typename T, unsigned long N>
	T SaintVenantElasticity::stress(const std::array<T, N> &strain, const int j) const
	{
		T res = elasticity_tensor_(j, 0)*strain[0];

		for(unsigned long k = 1; k < N; ++k)
			res += elasticity_tensor_(j, k)*strain[k];

		return res;
	}


	void SaintVenantElasticity::set_lambda_mu(const double lambda, const double mu)
	{
		elasticity_tensor_.set_from_lambda_mu(lambda, mu);
	}


	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	SaintVenantElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());


		if(size() == 2)
		{
			res(0) = (1./2*(((2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(1).getGradient()(1))+4*(pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(0).getGradient()(0))+(4*(pt(1).getGradient()(1))*elasticity_tensor_(1, 2)+(2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(1).getGradient()(0))+4*elasticity_tensor_(1, 2))*(pt(0).getGradient()(1))+(2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(1).getGradient()(1))+4*(pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2)))*(pt(1).getHessian()(1, 0))+(1./2*(6*(pt(0).getGradient()(0)*pt(0).getGradient()(0))*elasticity_tensor_(0, 2)+((4*elasticity_tensor_(0, 1)+8*elasticity_tensor_(2, 2))*(pt(0).getGradient()(1))+12*elasticity_tensor_(0, 2))*(pt(0).getGradient()(0))+6*(pt(0).getGradient()(1)*pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+(4*elasticity_tensor_(0, 1)+8*elasticity_tensor_(2, 2))*(pt(0).getGradient()(1))+2*(pt(1).getGradient()(1)*pt(1).getGradient()(1))*elasticity_tensor_(1, 2)+(4*(pt(1).getGradient()(0))*elasticity_tensor_(2, 2)+4*elasticity_tensor_(1, 2))*(pt(1).getGradient()(1))+2*(pt(1).getGradient()(0)*pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+4*(pt(1).getGradient()(0))*elasticity_tensor_(2, 2)+4*elasticity_tensor_(0, 2)))*(pt(0).getHessian()(1, 0))+(1./2*((elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(0).getGradient()(0)*pt(0).getGradient()(0))+(6*(pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+2*elasticity_tensor_(0, 1)+4*elasticity_tensor_(2, 2))*(pt(0).getGradient()(0))+3*elasticity_tensor_(1, 1)*(pt(0).getGradient()(1)*pt(0).getGradient()(1))+6*(pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+elasticity_tensor_(1, 1)*(pt(1).getGradient()(1)*pt(1).getGradient()(1))+(2*(pt(1).getGradient()(0))*elasticity_tensor_(1, 2)+2*elasticity_tensor_(1, 1))*(pt(1).getGradient()(1))+(pt(1).getGradient()(0)*pt(1).getGradient()(0))*elasticity_tensor_(0, 1)+2*(pt(1).getGradient()(0))*elasticity_tensor_(1, 2)+2*elasticity_tensor_(2, 2)))*(pt(0).getHessian()(1, 1))+(1./2*(3*elasticity_tensor_(0, 0)*(pt(0).getGradient()(0)*pt(0).getGradient()(0))+(6*(pt(0).getGradient()(1))*elasticity_tensor_(0, 2)+6*elasticity_tensor_(0, 0))*(pt(0).getGradient()(0))+(elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(0).getGradient()(1)*pt(0).getGradient()(1))+6*(pt(0).getGradient()(1))*elasticity_tensor_(0, 2)+elasticity_tensor_(0, 1)*(pt(1).getGradient()(1)*pt(1).getGradient()(1))+(2*(pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 1))*(pt(1).getGradient()(1))+elasticity_tensor_(0, 0)*(pt(1).getGradient()(0)*pt(1).getGradient()(0))+2*(pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 0)))*(pt(0).getHessian()(0, 0))+(1./2*((2*(pt(1).getGradient()(0))*elasticity_tensor_(0, 0)+2*(pt(1).getGradient()(1))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 2))*(pt(0).getGradient()(0))+(2*(pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+2*(pt(1).getGradient()(1))*elasticity_tensor_(2, 2)+2*elasticity_tensor_(2, 2))*(pt(0).getGradient()(1))+2*(pt(1).getGradient()(0))*elasticity_tensor_(0, 0)+2*(pt(1).getGradient()(1))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 2)))*(pt(1).getHessian()(0, 0))+(((pt(1).getGradient()(0))*elasticity_tensor_(2, 2)+(pt(1).getGradient()(1))*elasticity_tensor_(1, 2)+elasticity_tensor_(1, 2))*(pt(0).getGradient()(0))+((pt(1).getGradient()(0))*elasticity_tensor_(1, 2)+elasticity_tensor_(1, 1)*(pt(1).getGradient()(1))+elasticity_tensor_(1, 1))*(pt(0).getGradient()(1))+(pt(1).getGradient()(0))*elasticity_tensor_(2, 2)+(pt(1).getGradient()(1))*elasticity_tensor_(1, 2)+elasticity_tensor_(1, 2))*(pt(1).getHessian()(1, 1));
			res(1) = (1./2*(((2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(0).getGradient()(0))+4*(pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(1).getGradient()(1))+(4*(pt(0).getGradient()(0))*elasticity_tensor_(0, 2)+(2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(0).getGradient()(1))+4*elasticity_tensor_(0, 2))*(pt(1).getGradient()(0))+(2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(0).getGradient()(0))+4*(pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+2*elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2)))*(pt(0).getHessian()(1, 0))+(1./2*((elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(1).getGradient()(1)*pt(1).getGradient()(1))+(6*(pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 1)+4*elasticity_tensor_(2, 2))*(pt(1).getGradient()(1))+3*elasticity_tensor_(0, 0)*(pt(1).getGradient()(0)*pt(1).getGradient()(0))+6*(pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+elasticity_tensor_(0, 0)*(pt(0).getGradient()(0)*pt(0).getGradient()(0))+(2*(pt(0).getGradient()(1))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 0))*(pt(0).getGradient()(0))+(pt(0).getGradient()(1)*pt(0).getGradient()(1))*elasticity_tensor_(0, 1)+2*(pt(0).getGradient()(1))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(2, 2)))*(pt(1).getHessian()(0, 0))+(1./2*(6*(pt(1).getGradient()(1)*pt(1).getGradient()(1))*elasticity_tensor_(1, 2)+((4*elasticity_tensor_(0, 1)+8*elasticity_tensor_(2, 2))*(pt(1).getGradient()(0))+12*elasticity_tensor_(1, 2))*(pt(1).getGradient()(1))+6*(pt(1).getGradient()(0)*pt(1).getGradient()(0))*elasticity_tensor_(0, 2)+(4*elasticity_tensor_(0, 1)+8*elasticity_tensor_(2, 2))*(pt(1).getGradient()(0))+2*(pt(0).getGradient()(0)*pt(0).getGradient()(0))*elasticity_tensor_(0, 2)+(4*(pt(0).getGradient()(1))*elasticity_tensor_(2, 2)+4*elasticity_tensor_(0, 2))*(pt(0).getGradient()(0))+2*(pt(0).getGradient()(1)*pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+4*(pt(0).getGradient()(1))*elasticity_tensor_(2, 2)+4*elasticity_tensor_(1, 2)))*(pt(1).getHessian()(1, 0))+(1./2*(3*elasticity_tensor_(1, 1)*(pt(1).getGradient()(1)*pt(1).getGradient()(1))+(6*(pt(1).getGradient()(0))*elasticity_tensor_(1, 2)+6*elasticity_tensor_(1, 1))*(pt(1).getGradient()(1))+(elasticity_tensor_(0, 1)+2*elasticity_tensor_(2, 2))*(pt(1).getGradient()(0)*pt(1).getGradient()(0))+6*(pt(1).getGradient()(0))*elasticity_tensor_(1, 2)+(pt(0).getGradient()(0)*pt(0).getGradient()(0))*elasticity_tensor_(0, 1)+(2*(pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+2*elasticity_tensor_(0, 1))*(pt(0).getGradient()(0))+elasticity_tensor_(1, 1)*(pt(0).getGradient()(1)*pt(0).getGradient()(1))+2*(pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+2*elasticity_tensor_(1, 1)))*(pt(1).getHessian()(1, 1))+(1./2*((2*(pt(0).getGradient()(0))*elasticity_tensor_(0, 2)+2*(pt(0).getGradient()(1))*elasticity_tensor_(2, 2)+2*elasticity_tensor_(0, 2))*(pt(1).getGradient()(1))+(2*(pt(0).getGradient()(0))*elasticity_tensor_(0, 0)+2*(pt(0).getGradient()(1))*elasticity_tensor_(0, 2)+2*elasticity_tensor_(0, 0))*(pt(1).getGradient()(0))+2*(pt(0).getGradient()(0))*elasticity_tensor_(0, 2)+2*(pt(0).getGradient()(1))*elasticity_tensor_(2, 2)+2*elasticity_tensor_(0, 2)))*(pt(0).getHessian()(0, 0))+(pt(0).getHessian()(1, 1))*(((pt(0).getGradient()(0))*elasticity_tensor_(1, 2)+(pt(0).getGradient()(1))*elasticity_tensor_(1, 1)+elasticity_tensor_(1, 2))*(pt(1).getGradient()(1))+((pt(0).getGradient()(0))*elasticity_tensor_(2, 2)+(pt(0).getGradient()(1))*elasticity_tensor_(1, 2)+elasticity_tensor_(2, 2))*(pt(1).getGradient()(0))+(pt(0).getGradient()(0))*elasticity_tensor_(1, 2)+(pt(0).getGradient()(1))*elasticity_tensor_(1, 1)+elasticity_tensor_(1, 2));
		}
		else if(size() == 3)
		{
			res(0) = ((elasticity_tensor_(5, 5)+elasticity_tensor_(0, 1))*((pt(1).getGradient()(1)))+(elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(1).getGradient()(2)))+2*((pt(1).getGradient()(0)))*elasticity_tensor_(0, 5)+elasticity_tensor_(5, 5)+elasticity_tensor_(0, 1))*((pt(1).getHessian()(1,0)))+((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(1).getGradient()(1)))+(elasticity_tensor_(4, 4)+elasticity_tensor_(0, 2))*((pt(1).getGradient()(2)))+2*((pt(1).getGradient()(0)))*elasticity_tensor_(0, 4)+elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(1).getHessian()(2,0)))+((elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(1).getGradient()(1)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(1).getGradient()(2)))+2*((pt(1).getGradient()(0)))*elasticity_tensor_(4, 5)+elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(1).getHessian()(2,1)))+((elasticity_tensor_(5, 5)+elasticity_tensor_(0, 1))*((pt(2).getGradient()(1)))+(elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(2).getGradient()(2)))+2*((pt(2).getGradient()(0)))*elasticity_tensor_(0, 5)+elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(2).getHessian()(1,0)))+((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(2).getGradient()(1)))+(elasticity_tensor_(4, 4)+elasticity_tensor_(0, 2))*((pt(2).getGradient()(2)))+2*((pt(2).getGradient()(0)))*elasticity_tensor_(0, 4)+elasticity_tensor_(4, 4)+elasticity_tensor_(0, 2))*((pt(2).getHessian()(2,0)))+((elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(2).getGradient()(1)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(2).getGradient()(2)))+2*elasticity_tensor_(4, 5)*((pt(2).getGradient()(0)))+elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(2).getHessian()(2,1)))+((elasticity_tensor_(5, 5)+elasticity_tensor_(0, 1))*((pt(0).getGradient()(1)))+(elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(0).getGradient()(2)))+2*elasticity_tensor_(0, 5)*((pt(0).getGradient()(0))+1))*((pt(0).getHessian()(1,0)))+((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(0).getGradient()(1)))+(elasticity_tensor_(4, 4)+elasticity_tensor_(0, 2))*((pt(0).getGradient()(2)))+2*elasticity_tensor_(0, 4)*((pt(0).getGradient()(0))+1))*((pt(0).getHessian()(2,0)))+((elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(0).getGradient()(1)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(0).getGradient()(2)))+2*elasticity_tensor_(4, 5)*((pt(0).getGradient()(0))+1))*((pt(0).getHessian()(2,1)))+(((pt(0).getGradient()(1)))*elasticity_tensor_(0, 5)+((pt(0).getGradient()(2)))*elasticity_tensor_(0, 4)+elasticity_tensor_(0, 0)*((pt(0).getGradient()(0))+1))*((pt(0).getHessian()(0,0)))+(elasticity_tensor_(1, 5)*((pt(0).getGradient()(1)))+elasticity_tensor_(3, 5)*((pt(0).getGradient()(2)))+elasticity_tensor_(5, 5)*((pt(0).getGradient()(0))+1))*((pt(0).getHessian()(1,1)))+(elasticity_tensor_(3, 4)*((pt(0).getGradient()(1)))+elasticity_tensor_(2, 4)*((pt(0).getGradient()(2)))+elasticity_tensor_(4, 4)*((pt(0).getGradient()(0))+1))*((pt(0).getHessian()(2,2)))+(((pt(1).getGradient()(0)))*elasticity_tensor_(0, 0)+((pt(1).getGradient()(1)))*elasticity_tensor_(0, 5)+((pt(1).getGradient()(2)))*elasticity_tensor_(0, 4)+elasticity_tensor_(0, 5))*((pt(1).getHessian()(0,0)))+(elasticity_tensor_(1, 5)*((pt(1).getGradient()(1)))+elasticity_tensor_(3, 5)*((pt(1).getGradient()(2)))+elasticity_tensor_(5, 5)*((pt(1).getGradient()(0)))+elasticity_tensor_(1, 5))*((pt(1).getHessian()(1,1)))+(elasticity_tensor_(2, 4)*((pt(1).getGradient()(2)))+elasticity_tensor_(3, 4)*((pt(1).getGradient()(1)))+elasticity_tensor_(4, 4)*((pt(1).getGradient()(0)))+elasticity_tensor_(3, 4))*((pt(1).getHessian()(2,2)))+(((pt(2).getGradient()(0)))*elasticity_tensor_(0, 0)+((pt(2).getGradient()(1)))*elasticity_tensor_(0, 5)+((pt(2).getGradient()(2)))*elasticity_tensor_(0, 4)+elasticity_tensor_(0, 4))*((pt(2).getHessian()(0,0)))+(elasticity_tensor_(1, 5)*((pt(2).getGradient()(1)))+elasticity_tensor_(3, 5)*((pt(2).getGradient()(2)))+elasticity_tensor_(5, 5)*((pt(2).getGradient()(0)))+elasticity_tensor_(3, 5))*((pt(2).getHessian()(1,1)))+((pt(2).getHessian()(2,2)))*(elasticity_tensor_(2, 4)*((pt(2).getGradient()(2)))+elasticity_tensor_(3, 4)*((pt(2).getGradient()(1)))+elasticity_tensor_(4, 4)*((pt(2).getGradient()(0)))+elasticity_tensor_(2, 4));
			res(1) = ((elasticity_tensor_(5, 5)+elasticity_tensor_(0, 1))*((pt(0).getGradient()(0)))+(elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(0).getGradient()(2)))+2*elasticity_tensor_(1, 5)*((pt(0).getGradient()(1)))+elasticity_tensor_(5, 5)+elasticity_tensor_(0, 1))*((pt(0).getHessian()(1,0)))+((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(0).getGradient()(0)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(0).getGradient()(2)))+2*((pt(0).getGradient()(1)))*elasticity_tensor_(3, 5)+elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(0).getHessian()(2,0)))+((elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(0).getGradient()(0)))+(elasticity_tensor_(3, 3)+elasticity_tensor_(1, 2))*((pt(0).getGradient()(2)))+2*elasticity_tensor_(1, 3)*((pt(0).getGradient()(1)))+elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(0).getHessian()(2,1)))+((elasticity_tensor_(5, 5)+elasticity_tensor_(0, 1))*((pt(2).getGradient()(0)))+(elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(2).getGradient()(2)))+2*elasticity_tensor_(1, 5)*((pt(2).getGradient()(1)))+elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(2).getHessian()(1,0)))+((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(2).getGradient()(0)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(2).getGradient()(2)))+2*((pt(2).getGradient()(1)))*elasticity_tensor_(3, 5)+elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(2).getHessian()(2,0)))+((elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(2).getGradient()(0)))+(elasticity_tensor_(3, 3)+elasticity_tensor_(1, 2))*((pt(2).getGradient()(2)))+2*elasticity_tensor_(1, 3)*((pt(2).getGradient()(1)))+elasticity_tensor_(3, 3)+elasticity_tensor_(1, 2))*((pt(2).getHessian()(2,1)))+((elasticity_tensor_(5, 5)+elasticity_tensor_(0, 1))*((pt(1).getGradient()(0)))+(elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(1).getGradient()(2)))+2*elasticity_tensor_(1, 5)*((pt(1).getGradient()(1))+1))*((pt(1).getHessian()(1,0)))+((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(1).getGradient()(0)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(1).getGradient()(2)))+2*elasticity_tensor_(3, 5)*((pt(1).getGradient()(1))+1))*((pt(1).getHessian()(2,0)))+((elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(1).getGradient()(0)))+(elasticity_tensor_(3, 3)+elasticity_tensor_(1, 2))*((pt(1).getGradient()(2)))+2*elasticity_tensor_(1, 3)*((pt(1).getGradient()(1))+1))*((pt(1).getHessian()(2,1)))+(elasticity_tensor_(4, 5)*((pt(0).getGradient()(2)))+elasticity_tensor_(5, 5)*((pt(0).getGradient()(1)))+((pt(0).getGradient()(0)))*elasticity_tensor_(0, 5)+elasticity_tensor_(0, 5))*((pt(0).getHessian()(0,0)))+(elasticity_tensor_(1, 5)*((pt(0).getGradient()(0)))+((pt(0).getGradient()(1)))*elasticity_tensor_(1, 1)+((pt(0).getGradient()(2)))*elasticity_tensor_(1, 3)+elasticity_tensor_(1, 5))*((pt(0).getHessian()(1,1)))+(elasticity_tensor_(2, 3)*((pt(0).getGradient()(2)))+elasticity_tensor_(3, 3)*((pt(0).getGradient()(1)))+elasticity_tensor_(3, 4)*((pt(0).getGradient()(0)))+elasticity_tensor_(3, 4))*((pt(0).getHessian()(2,2)))+(((pt(1).getGradient()(0)))*elasticity_tensor_(0, 5)+elasticity_tensor_(4, 5)*((pt(1).getGradient()(2)))+elasticity_tensor_(5, 5)*((pt(1).getGradient()(1))+1))*((pt(1).getHessian()(0,0)))+(elasticity_tensor_(1, 5)*((pt(1).getGradient()(0)))+((pt(1).getGradient()(2)))*elasticity_tensor_(1, 3)+elasticity_tensor_(1, 1)*((pt(1).getGradient()(1))+1))*((pt(1).getHessian()(1,1)))+(elasticity_tensor_(3, 4)*((pt(1).getGradient()(0)))+elasticity_tensor_(2, 3)*((pt(1).getGradient()(2)))+elasticity_tensor_(3, 3)*((pt(1).getGradient()(1))+1))*((pt(1).getHessian()(2,2)))+(elasticity_tensor_(4, 5)*((pt(2).getGradient()(2)))+elasticity_tensor_(5, 5)*((pt(2).getGradient()(1)))+((pt(2).getGradient()(0)))*elasticity_tensor_(0, 5)+elasticity_tensor_(4, 5))*((pt(2).getHessian()(0,0)))+(elasticity_tensor_(1, 5)*((pt(2).getGradient()(0)))+((pt(2).getGradient()(1)))*elasticity_tensor_(1, 1)+((pt(2).getGradient()(2)))*elasticity_tensor_(1, 3)+elasticity_tensor_(1, 3))*((pt(2).getHessian()(1,1)))+((pt(2).getHessian()(2,2)))*(elasticity_tensor_(2, 3)*((pt(2).getGradient()(2)))+elasticity_tensor_(3, 3)*((pt(2).getGradient()(1)))+elasticity_tensor_(3, 4)*((pt(2).getGradient()(0)))+elasticity_tensor_(2, 3));
			res(2) = ((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(0).getGradient()(0)))+(elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(0).getGradient()(1)))+2*((pt(0).getGradient()(2)))*elasticity_tensor_(3, 4)+elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(0).getHessian()(1,0)))+((elasticity_tensor_(4, 4)+elasticity_tensor_(0, 2))*((pt(0).getGradient()(0)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(0).getGradient()(1)))+2*elasticity_tensor_(2, 4)*((pt(0).getGradient()(2)))+elasticity_tensor_(4, 4)+elasticity_tensor_(0, 2))*((pt(0).getHessian()(2,0)))+((elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(0).getGradient()(0)))+(elasticity_tensor_(3, 3)+elasticity_tensor_(1, 2))*((pt(0).getGradient()(1)))+2*elasticity_tensor_(2, 3)*((pt(0).getGradient()(2)))+elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(0).getHessian()(2,1)))+((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(1).getGradient()(0)))+(elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(1).getGradient()(1)))+2*((pt(1).getGradient()(2)))*elasticity_tensor_(3, 4)+elasticity_tensor_(3, 5)+elasticity_tensor_(1, 4))*((pt(1).getHessian()(1,0)))+((elasticity_tensor_(4, 4)+elasticity_tensor_(0, 2))*((pt(1).getGradient()(0)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(1).getGradient()(1)))+2*elasticity_tensor_(2, 4)*((pt(1).getGradient()(2)))+elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(1).getHessian()(2,0)))+((elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(1).getGradient()(0)))+(elasticity_tensor_(3, 3)+elasticity_tensor_(1, 2))*((pt(1).getGradient()(1)))+2*elasticity_tensor_(2, 3)*((pt(1).getGradient()(2)))+elasticity_tensor_(3, 3)+elasticity_tensor_(1, 2))*((pt(1).getHessian()(2,1)))+((elasticity_tensor_(4, 5)+elasticity_tensor_(0, 3))*((pt(2).getGradient()(0)))+(elasticity_tensor_(1, 4)+elasticity_tensor_(3, 5))*((pt(2).getGradient()(1)))+2*elasticity_tensor_(3, 4)*((pt(2).getGradient()(2))+1))*((pt(2).getHessian()(1,0)))+((elasticity_tensor_(4, 4)+elasticity_tensor_(0, 2))*((pt(2).getGradient()(0)))+(elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(2).getGradient()(1)))+2*elasticity_tensor_(2, 4)*((pt(2).getGradient()(2))+1))*((pt(2).getHessian()(2,0)))+((elasticity_tensor_(2, 5)+elasticity_tensor_(3, 4))*((pt(2).getGradient()(0)))+(elasticity_tensor_(3, 3)+elasticity_tensor_(1, 2))*((pt(2).getGradient()(1)))+2*elasticity_tensor_(2, 3)*((pt(2).getGradient()(2))+1))*((pt(2).getHessian()(2,1)))+(elasticity_tensor_(4, 4)*((pt(0).getGradient()(2)))+elasticity_tensor_(4, 5)*((pt(0).getGradient()(1)))+((pt(0).getGradient()(0)))*elasticity_tensor_(0, 4)+elasticity_tensor_(0, 4))*((pt(0).getHessian()(0,0)))+(elasticity_tensor_(3, 3)*((pt(0).getGradient()(2)))+((pt(0).getGradient()(0)))*elasticity_tensor_(3, 5)+elasticity_tensor_(1, 3)*((pt(0).getGradient()(1)))+elasticity_tensor_(3, 5))*((pt(0).getHessian()(1,1)))+(((pt(0).getGradient()(2)))*elasticity_tensor_(2, 2)+((pt(0).getGradient()(1)))*elasticity_tensor_(2, 3)+elasticity_tensor_(2, 4)*((pt(0).getGradient()(0)))+elasticity_tensor_(2, 4))*((pt(0).getHessian()(2,2)))+(elasticity_tensor_(4, 4)*((pt(1).getGradient()(2)))+elasticity_tensor_(4, 5)*((pt(1).getGradient()(1)))+((pt(1).getGradient()(0)))*elasticity_tensor_(0, 4)+elasticity_tensor_(4, 5))*((pt(1).getHessian()(0,0)))+(elasticity_tensor_(3, 3)*((pt(1).getGradient()(2)))+elasticity_tensor_(3, 5)*((pt(1).getGradient()(0)))+((pt(1).getGradient()(1)))*elasticity_tensor_(1, 3)+elasticity_tensor_(1, 3))*((pt(1).getHessian()(1,1)))+(((pt(1).getGradient()(2)))*elasticity_tensor_(2, 2)+elasticity_tensor_(2, 3)*((pt(1).getGradient()(1)))+elasticity_tensor_(2, 4)*((pt(1).getGradient()(0)))+elasticity_tensor_(2, 3))*((pt(1).getHessian()(2,2)))+(((pt(2).getGradient()(0)))*elasticity_tensor_(0, 4)+elasticity_tensor_(4, 5)*((pt(2).getGradient()(1)))+elasticity_tensor_(4, 4)*((pt(2).getGradient()(2))+1))*((pt(2).getHessian()(0,0)))+(elasticity_tensor_(3, 5)*((pt(2).getGradient()(0)))+elasticity_tensor_(1, 3)*((pt(2).getGradient()(1)))+elasticity_tensor_(3, 3)*((pt(2).getGradient()(2))+1))*((pt(2).getHessian()(1,1)))+((pt(2).getHessian()(2,2)))*(elasticity_tensor_(2, 4)*((pt(2).getGradient()(0)))+elasticity_tensor_(2, 3)*((pt(2).getGradient()(1)))+elasticity_tensor_(2, 2)*((pt(2).getGradient()(2))+1));
		}
		else
			assert(false);

		return res;
	}

	Eigen::VectorXd
	SaintVenantElasticity::assemble(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		// igl::Timer time; time.start();

		const int n_bases = vals.basis_values.size();

		Eigen::VectorXd grad;

		switch(size())
		{
			//2d
			case 2:
			{
				switch(n_bases)
				{
					//P1
					case 3:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 6, 1>>>(vals, displacement, da);
						grad = auto_diff_energy.getGradient();
						break;
					}
					//P2
					case 6:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 12, 1>>>(vals, displacement, da);
						grad = auto_diff_energy.getGradient();
						break;
					}
			 		//Q1
					case 4:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 8, 1>>>(vals, displacement, da);
						grad = auto_diff_energy.getGradient();
						break;
					}
					//Q2
					case 9:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 18, 1>>>(vals, displacement, da);
						grad = auto_diff_energy.getGradient();
						break;
					}
				}

				break;
			}


			//3d
			case 3:
			{
				switch(n_bases)
				{
					//P1
					case 4:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 12, 1>>>(vals, displacement, da);
						grad = auto_diff_energy.getGradient();
						break;
					}
					//P2
					case 10:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 30, 1>>>(vals, displacement, da);
						grad = auto_diff_energy.getGradient();
						break;
					}
					//Q1
					case 8:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 24, 1>>>(vals, displacement, da);
						grad = auto_diff_energy.getGradient();
						break;
					}
					//Q2
					case 27:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 81, 1>>>(vals, displacement, da);
						grad = auto_diff_energy.getGradient();
						break;
					}
				}
			}
		}


		if(grad.size()<=0)
		{
			static bool show_message = true;

			if(show_message)
			{
				std::cout<<"[Warning] "<<n_bases<<" not using static sizes"<<std::endl;
				show_message = false;
			}

			auto auto_diff_energy = compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(vals, displacement, da);
			grad = auto_diff_energy.getGradient();
		}

		// time.stop();
		// std::cout << "-- grad: " << time.getElapsedTime() << std::endl;

		return grad;
	}

	Eigen::MatrixXd
	SaintVenantElasticity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		// igl::Timer time; time.start();

		const int n_bases = vals.basis_values.size();
		Eigen::MatrixXd hessian;

		switch(size())
		{
			//2d
			case 2:
			{
				switch(n_bases)
				{
					//P1
					case 3:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>>(vals, displacement, da);
						hessian = auto_diff_energy.getHessian();
						break;
					}
					//P2
					case 6:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>>(vals, displacement, da);
						hessian = auto_diff_energy.getHessian();
						break;
					}
					//Q1
					case 4:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>>(vals, displacement, da);
						hessian = auto_diff_energy.getHessian();
						break;
					}
					//Q2
					case 9:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>>(vals, displacement, da);
						hessian = auto_diff_energy.getHessian();
						break;
					}
				}

				break;
			}


			//3d
			case 3:
			{
				switch(n_bases)
				{
					//P2
					case 4:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>>(vals, displacement, da);
						hessian = auto_diff_energy.getHessian();
						break;
					}
					//P2
					case 10:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>>(vals, displacement, da);
						hessian = auto_diff_energy.getHessian();
						break;
					}
			 		//Q1
					case 8:
					{
						auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>>(vals, displacement, da);
						hessian = auto_diff_energy.getHessian();
						break;
					}

					// // Q2
					// case 27:
					// {
					// 	auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>>(vals, displacement, da);
					// 	hessian = auto_diff_energy.getHessian();
					// 	break;
					// }
				}
			}
		}

		if(hessian.size() <= 0)
		{
			static bool show_message = true;

			if(show_message)
			{
				std::cout<<"[Warning] "<<n_bases<<" not using static sizes"<<std::endl;
				show_message = false;
			}

			auto auto_diff_energy = compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(vals, displacement, da);
			hessian = auto_diff_energy.getHessian();
		}

		// time.stop();
		// std::cout << "-- hessian: " << time.getElapsedTime() << std::endl;

		return hessian;
	}

	void SaintVenantElasticity::compute_von_mises_stresses(const ElementBases &bs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);

		stresses.resize(local_pts.rows(), 1);

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, bs);


		for(long p = 0; p < local_pts.rows(); ++p)
		{
			displacement_grad.setZero();

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.grad.rows() == local_pts.rows());
				assert(loc_val.grad.cols() == size());

				for(int d = 0; d < size(); ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						displacement_grad.row(d) += b.global()[ii].val * loc_val.grad.row(p) * displacement(b.global()[ii].index*size() + d);
					}
				}
			}

			displacement_grad = displacement_grad * vals.jac_it[p];

			Eigen::MatrixXd strain = strain_from_disp_grad(displacement_grad);
			Eigen::MatrixXd stress_tensor(size(), size());

			if(size() == 2)
			{
				std::array<double, 3> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = 2*strain(0,1);


				stress_tensor <<
				stress(eps, 0), stress(eps, 2),
				stress(eps, 2), stress(eps, 1);
			}
			else
			{
				std::array<double, 6> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = strain(2,2);
				eps[3] = 2*strain(1,2);
				eps[4] = 2*strain(0,2);
				eps[5] = 2*strain(0,1);

				stress_tensor <<
				stress(eps, 0), stress(eps, 5), stress(eps, 4),
				stress(eps, 5), stress(eps, 1), stress(eps, 3),
				stress(eps, 4), stress(eps, 3), stress(eps, 2);
			}

			stresses(p) = von_mises_stress_for_stress_tensor(stress_tensor);
		}
	}

	double SaintVenantElasticity::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		return compute_energy_aux<double>(vals, displacement, da);
	}

	//Compute \int \sigma : E
	template<typename T>
	T SaintVenantElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> 						AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> 	AutoDiffGradMat;

		assert(displacement.cols() == 1);

		const int n_pts = da.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(vals.basis_values.size() * size(), 1);
		local_dispv.setZero();
		for(size_t i = 0; i < vals.basis_values.size(); ++i){
			const auto &bs = vals.basis_values[i];
			for(size_t ii = 0; ii < bs.global.size(); ++ii){
				for(int d = 0; d < size(); ++d){
					local_dispv(i*size() + d) += bs.global[ii].val * displacement(bs.global[ii].index*size() + d);
				}
			}
		}

		DiffScalarBase::setVariableCount(local_dispv.rows());
		AutoDiffVect local_disp(local_dispv.rows(), 1);
		T energy = T(0.0);

		const AutoDiffAllocator<T> allocate_auto_diff_scalar;

		for(long i = 0; i < local_dispv.rows(); ++i){
			local_disp(i) = allocate_auto_diff_scalar(i, local_dispv(i));
		}

		AutoDiffGradMat displacement_grad(size(), size());

		for(long p = 0; p < n_pts; ++p)
		{
			bool is_disp_grad_set = false;

			for(size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				const Eigen::MatrixXd grad = bs.grad*vals.jac_it[p];
				assert(grad.cols() == size());
				assert(size_t(grad.rows()) ==  vals.jac_it.size());

				for(int d = 0; d < size(); ++d)
				{
					for(int c = 0; c < size(); ++c)
					{
						if(is_disp_grad_set)
							displacement_grad(d, c) += grad(p, c) * local_disp(i*size() + d);
						else
							displacement_grad(d, c) = grad(p, c) * local_disp(i*size() + d);
					}
				}

				is_disp_grad_set = true;
			}

			// displacement_grad = displacement_grad * vals.jac_it[p];

			AutoDiffGradMat strain = strain_from_disp_grad(displacement_grad);
			AutoDiffGradMat stress_tensor(size(), size());

			if(size() == 2)
			{
				std::array<T, 3> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = 2*strain(0,1);


				stress_tensor <<
				stress(eps, 0), stress(eps, 2),
				stress(eps, 2), stress(eps, 1);
			}
			else
			{
				std::array<T, 6> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = strain(2,2);
				eps[3] = 2*strain(1,2);
				eps[4] = 2*strain(0,2);
				eps[5] = 2*strain(0,1);

				stress_tensor <<
				stress(eps, 0), stress(eps, 5), stress(eps, 4),
				stress(eps, 5), stress(eps, 1), stress(eps, 3),
				stress(eps, 4), stress(eps, 3), stress(eps, 2);
			}

			energy += (stress_tensor * strain).trace() * da(p);
		}

		return energy * 0.5;
	}
}
