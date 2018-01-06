#include "Harmonic.hpp"
#include "PolygonQuadrature.hpp"

#include <iostream>
#include <fstream>

namespace poly_fem
{
	namespace
	{
		double kernel(const double r)
		{
			if(r < 1e-8) return 0;

			return log(r);
		}

		double kernel_prime(const double r)
		{
			if(r < 1e-8) return 0;

			return 1/r;
		}
	}

	Harmonic::Harmonic(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples, const Eigen::MatrixXd &local_basis_integral, Eigen::MatrixXd &rhs)
	: centers_(centers)
	{
		compute(samples, local_basis_integral, rhs);
	}


	void Harmonic::basis(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		const Eigen::MatrixXd w = weights_.col(local_index);
		const long end = w.size()-1;

		val.resize(uv.rows(), 1);
		val.setConstant(w(end)); //constant term

		for(long i = 0; i < uv.rows(); ++i)
		{
			const Eigen::MatrixXd p = uv.row(i);
			for(long k = 0; k < centers_.rows(); ++k)
				val(i) += w(k) * kernel((centers_.row(k)-p).norm());

			val(i) += p(0) * w(end - 2) + p(1)* w(end - 1);	//linear term
		}
	}

	void Harmonic::grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		const Eigen::MatrixXd w = weights_.col(local_index);
		const long end = w.size()-1;

		val.resize(uv.rows(), 2);

		for(long i = 0; i < uv.rows(); ++i)
		{
			val(i, 0) = w(end-2);
			val(i, 1) = w(end-1);

			const auto &p = uv.row(i);
			for(long k = 0; k < centers_.rows(); ++k)
			{
				const double r = (centers_.row(k)-p).norm();

				val(i, 0) += w(k)*(p(0)-centers_(k,0))*kernel_prime(r)/r;
				val(i, 1) += w(k)*(p(1)-centers_(k,1))*kernel_prime(r)/r;
			}
		}
	}



	void Harmonic::compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &local_basis_integral, Eigen::MatrixXd &rhs)
	{
		// const int size = (int) samples.rows();

		// //+2 linear, +1 constant
		// Eigen::MatrixXd mat(size, centers_.rows() + 2 + 1);

		// const int end = int(mat.cols())-1;
		// mat.col(end).setOnes();

		// for(long i = 0; i < samples.rows(); ++i)
		// {
		// 	mat(i, end - 2) = samples(i, 0);
		// 	mat(i, end - 1) = samples(i, 1);

		// 	for(long j = 0; j < centers_.rows(); ++j)
		// 	{
		// 		mat(i,j)=kernel((centers_.row(j)-samples.row(i)).norm());
		// 	}
		// }

		// weights_ = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);


		const int size = (int) samples.rows();

		//+1 constant
		Eigen::MatrixXd mat(size, centers_.rows() + 1);

		const int end = int(mat.cols())-1;
		mat.col(end).setOnes();

		Quadrature quad;
		PolygonQuadrature p_quad;
		p_quad.get_quadrature(samples, 4, quad);

		for(long i = 0; i < samples.rows(); ++i)
		{
			const double x = samples(i, 0);
			const double y = samples(i, 1);

			rhs.row(i) -= (local_basis_integral.col(0).transpose() * x + local_basis_integral.col(1).transpose() * y)/quad.weights.sum();

			for(long j = 0; j < centers_.rows(); ++j)
			{
				const double r = (centers_.row(j)-samples.row(i)).norm();
				const Eigen::MatrixXd diff_r_x = quad.points.col(0).array() - centers_(j, 0);
				const Eigen::MatrixXd diff_r_y = quad.points.col(1).array() - centers_(j, 1);

				const Eigen::MatrixXd rr = (diff_r_x.array() * diff_r_x.array() + diff_r_y.array() *diff_r_y.array());

				const double KxI = (diff_r_x.array() / rr.array() * quad.weights.array()).sum()/quad.weights.sum();
				const double KyI = (diff_r_y.array() / rr.array() * quad.weights.array()).sum()/quad.weights.sum();

				mat(i,j)=kernel(r) - x*KxI - y*KyI;
			}
		}

		Eigen::MatrixXd tmp = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
		weights_.resize(centers_.rows() + 2 + 1, tmp.cols());
		weights_.block(0, 0, centers_.rows(), tmp.cols()) = tmp.block(0, 0, centers_.rows(), tmp.cols());

		const int wend = weights_.rows()-1;

		weights_.row(wend) = tmp.row(tmp.rows()-1);
		weights_.row(wend-2) = local_basis_integral.col(0).transpose()/quad.weights.sum();
		weights_.row(wend-1) = local_basis_integral.col(1).transpose()/quad.weights.sum();

		for(long j = 0; j < centers_.rows(); ++j)
		{
			const Eigen::MatrixXd diff_r_x = quad.points.col(0).array() - centers_(j, 0);
			const Eigen::MatrixXd diff_r_y = quad.points.col(1).array() - centers_(j, 1);

			const Eigen::MatrixXd rr = (diff_r_x.array() * diff_r_x.array() + diff_r_y.array() *diff_r_y.array());

			const double KxI = (diff_r_x.array() / rr.array() * quad.weights.array()).sum()/quad.weights.sum();
			const double KyI = (diff_r_y.array() / rr.array() * quad.weights.array()).sum()/quad.weights.sum();

			weights_.row(wend-2) -= weights_.row(j)*KxI;
			weights_.row(wend-1) -= weights_.row(j)*KyI;
		}

		Eigen::MatrixXd integralx = Eigen::MatrixXd::Zero(weights_.cols(), 1);
		Eigen::MatrixXd integraly = Eigen::MatrixXd::Zero(weights_.cols(), 1);
		Eigen::MatrixXd gradv;
		for(long asd = 0; asd < quad.weights.rows(); ++asd)
		{
			for(long k = 0; k < weights_.cols(); ++k){
				grad(k, quad.points.row(asd), gradv);
				integralx(k) += gradv(0)*quad.weights(asd);
				integraly(k) += gradv(1)*quad.weights(asd);
			}
		}

		// std::cout<<integralx-local_basis_integral.col(0)<<"\n------------\n"<<std::endl;
		// std::cout<<integraly-local_basis_integral.col(1)<<"\n------------\n"<<std::endl;
		// std::cout<<local_basis_integral<<"\n------------\n"<<std::endl;

		// std::cout.precision(100);
		// // std::cout<<"mat=[\n"<<mat<<"];"<<std::endl;
		// std::cout<<"weights=[\n"<<tmp<<"];"<<std::endl;
		// std::cout<<"weights=[\n"<<weights_<<"];"<<std::endl;

		// {
		// 	std::ofstream os;
		// 	os.open("cc.txt");
		// 	os.precision(100);
		// 	os<<centers_<<std::endl;
		// 	os.close();
		// }

		// {
		// 	std::ofstream os;
		// 	os.open("ss.txt");
		// 	os.precision(100);
		// 	os<<samples<<std::endl;
		// 	os.close();
		// }

		// // {
		// // 	std::ofstream os;
		// // 	os.open("mat.txt");
		// // 	os.precision(100);
		// // 	os<<mat<<std::endl;
		// // 	os.close();
		// // }

		// {
		// 	std::ofstream os;
		// 	os.open("rr.txt");
		// 	os.precision(100);
		// 	os<<rhs<<std::endl;
		// 	os.close();
		// }

		// {
		// 	std::ofstream os;
		// 	os.open("ww.txt");
		// 	os.precision(100);
		// 	os<<weights_<<std::endl;
		// 	os.close();
		// }
	}
}
