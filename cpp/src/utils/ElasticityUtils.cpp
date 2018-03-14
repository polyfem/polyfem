#include "ElasticityUtils.hpp"


namespace poly_fem
{
	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress)
	{
		double von_mises_stress =  0.5 * ( stress(0, 0) - stress(1, 1) ) * ( stress(0, 0) - stress(1, 1) ) + 3.0  *  stress(0, 1) * stress(0, 1);

		if(stress.rows() == 3)
		{
			von_mises_stress += 0.5 * (stress(2, 2) - stress(1, 1)) * (stress(2, 2) - stress(1, 1)) + 3.0  * stress(2, 1) * stress(2, 1);
			von_mises_stress += 0.5 * (stress(2, 2) - stress(0, 0)) * (stress(2, 2) - stress(0, 0)) + 3.0  * stress(2, 0) * stress(2, 0);
		}

		von_mises_stress = sqrt( fabs(von_mises_stress) );

		return von_mises_stress;
	}



	void ElasticityTensor::resize(const int size)
	{
		if(size == 2)
			stifness_tensor_.resize(6, 1);
		else
			stifness_tensor_.resize(21, 1);

		stifness_tensor_.setZero();

		size_ = size;
	}

	double ElasticityTensor::operator()(int i, int j) const
	{
		if(j < i)
		{
			int tmp=i;
			i = j;
			j = tmp;
		}

		assert(j>=i);
		const int n = size_ == 2 ? 3 : 6;
		assert(i < n);
		assert(j < n);
		assert(i >= 0);
		assert(j >= 0);
		const int index = n * i + j - i * (i + 1) / 2;
		assert(index < stifness_tensor_.size());

		return stifness_tensor_(index);
	}

	double &ElasticityTensor::operator()(int i, int j)
	{
		if(j < i)
		{
			int tmp=i;
			i = j;
			j = tmp;
		}

		assert(j>=i);
		const int n = size_ == 2 ? 3 : 6;
		assert(i < n);
		assert(j < n);
		assert(i >= 0);
		assert(j >= 0);
		const int index = n * i + j - i * (i + 1) / 2;
		assert(index < stifness_tensor_.size());

		return stifness_tensor_(index);
	}


	void ElasticityTensor::set_from_entries(const std::vector<double> &entries)
	{
		if(size_ == 2)
		{
			assert(entries.size() >= 6);

			(*this)(0, 0) = entries[0];
			(*this)(0, 1) = entries[1];
			(*this)(0, 2) = entries[2];

			(*this)(1, 1) = entries[3];
			(*this)(1, 2) = entries[4];

			(*this)(2, 2) = entries[5];
		}
		else
		{
			assert(entries.size() >= 21);

			(*this)(0, 0) = entries[0];
			(*this)(0, 1) = entries[1];
			(*this)(0, 2) = entries[2];
			(*this)(0, 3) = entries[3];
			(*this)(0, 4) = entries[4];
			(*this)(0, 5) = entries[5];

			(*this)(1, 1) = entries[6];
			(*this)(1, 2) = entries[7];
			(*this)(1, 3) = entries[8];
			(*this)(1, 4) = entries[9];
			(*this)(1, 5) = entries[10];

			(*this)(2, 2) = entries[11];
			(*this)(2, 3) = entries[12];
			(*this)(2, 4) = entries[13];
			(*this)(2, 5) = entries[14];

			(*this)(3, 3) = entries[15];
			(*this)(3, 4) = entries[16];
			(*this)(3, 5) = entries[17];

			(*this)(4, 4) = entries[18];
			(*this)(4, 5) = entries[19];

			(*this)(5, 5) = entries[20];

		}
	}


	void ElasticityTensor::set_from_lambda_mu(const double lambda, const double mu)
	{
		if(size_ == 2)
		{
			(*this)(0, 0) = 2*mu+lambda;
			(*this)(0, 1) = lambda;
			(*this)(0, 2) = 0;

			(*this)(1, 1) = 2*mu+lambda;
			(*this)(1, 2) = 0;

			(*this)(2, 2) = mu;
		}
		else
		{
			(*this)(0, 0) = 2*mu+lambda;
			(*this)(0, 1) = lambda;
			(*this)(0, 2) = lambda;
			(*this)(0, 3) = 0;
			(*this)(0, 4) = 0;
			(*this)(0, 5) = 0;

			(*this)(1, 1) = 2*mu+lambda;
			(*this)(1, 2) = lambda;
			(*this)(1, 3) = 0;
			(*this)(1, 4) = 0;
			(*this)(1, 5) = 0;

			(*this)(2, 2) = 2*mu+lambda;
			(*this)(2, 3) = 0;
			(*this)(2, 4) = 0;
			(*this)(2, 5) = 0;

			(*this)(3, 3) = mu;
			(*this)(3, 4) = 0;
			(*this)(3, 5) = 0;

			(*this)(4, 4) = mu;
			(*this)(4, 5) = 0;

			(*this)(5, 5) = mu;

		}
	}

	template<int DIM>
	double ElasticityTensor::compute_stress(const std::array<double, DIM> &strain, const int j) const
	{
		double res = 0;

		for(int k = 0; k < DIM; ++k)
			res += (*this)(j, k)*strain[k];

		return res;
	}


	//template instantiation
	template double ElasticityTensor::compute_stress<3>(const std::array<double, 3> &strain, const int j) const;
	template double ElasticityTensor::compute_stress<6>(const std::array<double, 6> &strain, const int j) const;

}