#include "Harmonic.hpp"
#include "PolygonQuadrature.hpp"

#include <iostream>
#include <fstream>

namespace poly_fem
{
	namespace
	{
		double kernel(const bool is_volume, const double r)
		{
			if(r < 1e-8) return 0;

			if(is_volume)
				return -1/r;

			return log(r);
		}

		double kernel_prime(const bool is_volume, const double r)
		{
			if(r < 1e-8) return 0;

			if(is_volume)
				return 1/(r*r);

			return 1/r;
		}
	}

	Harmonic::Harmonic(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples,
		const Eigen::MatrixXd &local_basis_integral, const Quadrature &quadr, Eigen::MatrixXd &rhs)
	: centers_(centers)
	{
		compute(samples, local_basis_integral, quadr, rhs);
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
				val(i) += w(k) * kernel(is_volume_, (centers_.row(k)-p).norm());

			//linear term
			if(is_volume_)
				val(i) += p(0) * w(end - 3) + p(1) * w(end - 2) + p(2) * w(end - 1);
			else
				val(i) += p(0) * w(end - 2) + p(1) * w(end - 1);
		}
	}

	void Harmonic::grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		const Eigen::MatrixXd w = weights_.col(local_index);
		const long end = w.size()-1;

		if(is_volume_)
			val.resize(uv.rows(), 3);
		else
			val.resize(uv.rows(), 2);

		for(long i = 0; i < uv.rows(); ++i)
		{
			if(is_volume_)
			{
				val(i, 0) = w(end-3);
				val(i, 1) = w(end-2);
				val(i, 2) = w(end-1);
			}
			else
			{
				val(i, 0) = w(end-2);
				val(i, 1) = w(end-1);
			}

			const auto &p = uv.row(i);
			for(long k = 0; k < centers_.rows(); ++k)
			{
				const double r = (centers_.row(k)-p).norm();

				val(i, 0) += w(k)*(p(0)-centers_(k,0))*kernel_prime(is_volume_, r)/r;
				val(i, 1) += w(k)*(p(1)-centers_(k,1))*kernel_prime(is_volume_, r)/r;

				if(is_volume_)
					val(i, 2) += w(k)*(p(2)-centers_(k,2))*kernel_prime(is_volume_, r)/r;
			}
		}
	}



	void Harmonic::compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &local_basis_integral,
		const Quadrature &quad, Eigen::MatrixXd &rhs)
	{
		is_volume_ = samples.cols() == 3;

#if 0
		const int size = (int) samples.rows();

		//+2 linear, +1 constant
		Eigen::MatrixXd mat(size, centers_.rows() + 2 + 1);

		const int end = int(mat.cols())-1;
		mat.col(end).setOnes();

		for(long i = 0; i < samples.rows(); ++i)
		{
			mat(i, end - 2) = samples(i, 0);
			mat(i, end - 1) = samples(i, 1);

			for(long j = 0; j < centers_.rows(); ++j)
			{
				mat(i,j)=kernel(is_volume_, (centers_.row(j)-samples.row(i)).norm());
			}
		}

		weights_ = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
#else
		const int size = (int) samples.rows();

		//+1 constant
		Eigen::MatrixXd mat(size, centers_.rows() + 1);

		const int end = int(mat.cols())-1;
		mat.col(end).setOnes();

		for(long i = 0; i < samples.rows(); ++i)
		{
			const double x = samples(i, 0);
			const double y = samples(i, 1);
			const double z = is_volume_ ? samples(i, 2) : 0;

			rhs.row(i) -= (local_basis_integral.col(0).transpose() * x + local_basis_integral.col(1).transpose() * y)/quad.weights.sum();
			if(is_volume_)
				rhs.row(i) -= local_basis_integral.col(2).transpose() * z / quad.weights.sum();

			for(long j = 0; j < centers_.rows(); ++j)
			{
				const double r = (centers_.row(j)-samples.row(i)).norm();
				const Eigen::MatrixXd diff_r_x = quad.points.col(0).array() - centers_(j, 0);
				const Eigen::MatrixXd diff_r_y = quad.points.col(1).array() - centers_(j, 1);
				Eigen::MatrixXd diff_r_z;
				if(is_volume_)
					diff_r_z = quad.points.col(2).array() - centers_(j, 2);

				const Eigen::MatrixXd rr = is_volume_ ?
				(diff_r_x.array() * diff_r_x.array() + diff_r_y.array() * diff_r_y.array() + diff_r_z.array() * diff_r_z.array()).sqrt().eval() :
				(diff_r_x.array() * diff_r_x.array() + diff_r_y.array() * diff_r_y.array()).sqrt().eval();

				double KxI = 0;
				double KyI = 0;
				double KzI = 0;

				for(long k = 0; k < rr.size(); ++k)
				{
					KxI += diff_r_x(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quad.weights(k);
					KyI += diff_r_y(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quad.weights(k);

					if(is_volume_)
						KzI += diff_r_z(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quad.weights(k);
				}

				KxI /= quad.weights.sum();
				KyI /= quad.weights.sum();
				KzI /= quad.weights.sum();

				mat(i,j)=kernel(is_volume_, r) - x*KxI - y*KyI - KzI*z;
			}
		}

		Eigen::MatrixXd tmp = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
		weights_.resize(centers_.rows() + (is_volume_?3:2) + 1, tmp.cols());
		weights_.block(0, 0, centers_.rows(), tmp.cols()) = tmp.block(0, 0, centers_.rows(), tmp.cols());

		const int wend = weights_.rows()-1;

		weights_.row(wend) = tmp.row(tmp.rows()-1);
		if(is_volume_)
		{
			weights_.row(wend-3) = local_basis_integral.col(0).transpose()/quad.weights.sum();
			weights_.row(wend-2) = local_basis_integral.col(1).transpose()/quad.weights.sum();
			weights_.row(wend-1) = local_basis_integral.col(2).transpose()/quad.weights.sum();
		}
		else
		{
			weights_.row(wend-2) = local_basis_integral.col(0).transpose()/quad.weights.sum();
			weights_.row(wend-1) = local_basis_integral.col(1).transpose()/quad.weights.sum();
		}

		for(long j = 0; j < centers_.rows(); ++j)
		{
			const Eigen::MatrixXd diff_r_x = quad.points.col(0).array() - centers_(j, 0);
			const Eigen::MatrixXd diff_r_y = quad.points.col(1).array() - centers_(j, 1);
			Eigen::MatrixXd diff_r_z;
			if(is_volume_)
				diff_r_z = quad.points.col(2).array() - centers_(j, 2);

			const Eigen::MatrixXd rr = is_volume_ ?
			(diff_r_x.array() * diff_r_x.array() + diff_r_y.array() * diff_r_y.array() + diff_r_z.array() * diff_r_z.array()).sqrt().eval() :
			(diff_r_x.array() * diff_r_x.array() + diff_r_y.array() * diff_r_y.array()).sqrt().eval();

			double KxI = 0;
			double KyI = 0;
			double KzI = 0;

			for(long k = 0; k < rr.size(); ++k)
			{
				KxI += diff_r_x(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quad.weights(k);
				KyI += diff_r_y(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quad.weights(k);

				if(is_volume_)
					KzI += diff_r_z(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quad.weights(k);
			}

			KxI /= quad.weights.sum();
			KyI /= quad.weights.sum();
			KzI /= quad.weights.sum();

			if(is_volume_)
			{
				weights_.row(wend-3) -= weights_.row(j)*KxI;
				weights_.row(wend-2) -= weights_.row(j)*KyI;
				weights_.row(wend-1) -= weights_.row(j)*KzI;
			}
			else
			{
				weights_.row(wend-2) -= weights_.row(j)*KxI;
				weights_.row(wend-1) -= weights_.row(j)*KyI;
			}
		}

		// Eigen::MatrixXd integralx = Eigen::MatrixXd::Zero(weights_.cols(), 1);
		// Eigen::MatrixXd integraly = Eigen::MatrixXd::Zero(weights_.cols(), 1);
		// Eigen::MatrixXd gradv;
		// for(long asd = 0; asd < quad.weights.rows(); ++asd)
		// {
		// 	for(long k = 0; k < weights_.cols(); ++k){
		// 		grad(k, quad.points.row(asd), gradv);
		// 		integralx(k) += gradv(0)*quad.weights(asd);
		// 		integraly(k) += gradv(1)*quad.weights(asd);
		// 	}
		// }

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
#endif
	}
}
