#include "BiharmonicBasis.hpp"

#include <iostream>

namespace poly_fem
{
	namespace
	{
		double kernel(const double r)
		{
			if(r < 1e-8) return 0;

			return r * r * (log(r)-1);
		}

		double kernel_prime(const double r)
		{
			if(r < 1e-8) return 0;

			return r * ( 2 * log(r) - 1);
		}
	}

	BiharmonicBasis::BiharmonicBasis(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples, const Eigen::MatrixXd &rhs)
	: centers_(centers)
	{
		compute(samples, rhs);
	}


	void BiharmonicBasis::basis(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
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
			val(i) += w(end - 5)*p(0)*p(0) + 2*w(end - 4)*p(0)*p(1) + w(end - 3)*p(1)*p(1);
		}
	}

	void BiharmonicBasis::grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
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

			val(i, 0) += 2 * w(end-5)*p(0) + 2 * w(end-4)*p(1);
			val(i, 1) += 2 * w(end-3)*p(1) + 2 * w(end-4)*p(0);
		}
	}



	void BiharmonicBasis::compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &rhs)
	{
		const int offset = (int) samples.rows();

		const bool impose_c1 = rhs.rows() != offset;

		const int size = offset + (impose_c1?(2*offset):0);

		assert(size == rhs.rows());

		//+3 bilinear, +2 linear, +1 constant
		Eigen::MatrixXd mat(size, centers_.rows() + 3 + 2 + 1);

		const int end = int(mat.cols())-1;
		mat.col(end).setOnes();

		for(long i = 0; i < samples.rows(); ++i)
		{
			mat(i, end - 1) = samples(i, 1);
			mat(i, end - 2) = samples(i, 0);

			mat(i, end - 3) = samples(i, 1) * samples(i, 1);
			mat(i, end - 4) = 2 * samples(i, 0) * samples(i, 1);
			mat(i, end - 5) = samples(i, 0)*samples(i, 0);

			for(long j = 0; j < centers_.rows(); ++j)
			{
				mat(i,j)=kernel((centers_.row(j)-samples.row(i)).norm());
			}
		}

		if(impose_c1)
		{
			mat.block(offset, end, 2*offset, 1).setZero();
			mat.block(offset, end-1, 2*offset, 1).setZero();
			mat.block(offset, end-2, 2*offset, 1).setZero();

			for(long i = 0; i < samples.rows(); ++i)
			{
				const auto &p = samples.row(i);

				mat(offset + 2*i,   end - 1) = 0;
				mat(offset + 2*i+1, end - 1) = 1;

				mat(offset + 2*i,   end - 2) = 1;
				mat(offset + 2*i+1, end - 2) = 0;

				mat(offset + 2*i,   end - 3) = 0;
				mat(offset + 2*i+1, end - 3) = 2*p(1);

				mat(offset + 2*i,   end - 4) = 2*p(1);
				mat(offset + 2*i+1, end - 4) = 2*p(0);

				mat(offset + 2*i,   end - 5) = 2*p(0);
				mat(offset + 2*i+1, end - 5) = 0;


				for(long j = 0; j < centers_.rows(); ++j)
				{
					const double r = (centers_.row(j)-p).norm();

					mat(offset + 2*i,   j) = (p(0)-centers_(j,0))*kernel_prime(r)/r;
					mat(offset + 2*i+1, j) = (p(1)-centers_(j,1))*kernel_prime(r)/r;
				}
			}
		}

		weights_ = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);

		// std::cout.precision(100);
		// std::cout<<"cc=[\n"<<centers_<<"];"<<std::endl;
		// std::cout<<"ss=[\n"<<samples<<"];"<<std::endl;
		// std::cout<<"mat=[\n"<<mat<<"];"<<std::endl;
		// std::cout<<"rr=[\n"<<rhs<<"];"<<std::endl;
		// std::cout<<"ww=[\n"<<weights_<<"];"<<std::endl;
	}
}
