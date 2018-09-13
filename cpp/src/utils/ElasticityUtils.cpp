#include <polyfem/ElasticityUtils.hpp>
#include <polyfem/Logger.hpp>

namespace polyfem
{
	Eigen::VectorXd gradient_from_energy(const int size, const int n_bases, const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da,
		const std::function<DScalar1<double, Eigen::Matrix<double, 6, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun6,
		const std::function<DScalar1<double, Eigen::Matrix<double, 8, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun8,
		const std::function<DScalar1<double, Eigen::Matrix<double, 12, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun12,
		const std::function<DScalar1<double, Eigen::Matrix<double, 18, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun18,
		const std::function<DScalar1<double, Eigen::Matrix<double, 24, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun24,
		const std::function<DScalar1<double, Eigen::Matrix<double, 30, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun30,
		const std::function<DScalar1<double, Eigen::Matrix<double, 60, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun60,
		const std::function<DScalar1<double, Eigen::Matrix<double, 81, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun81,
		const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funN,
		const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, BIG_N, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funBigN,
		const std::function<DScalar1<double, Eigen::VectorXd>				(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funn
		)
	{
		Eigen::VectorXd grad;

		switch(size * n_bases)
		{
			case 6:
			{
				auto auto_diff_energy = fun6(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
				break;
			}
			case 8:
			{
				auto auto_diff_energy = fun8(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
				break;
			}
			case 12:
			{
				auto auto_diff_energy = fun12(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
				break;
			}
			case 18:
			{
				auto auto_diff_energy = fun18(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
				break;
			}
			case 24:
			{
				auto auto_diff_energy = fun24(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
				break;
			}
			case 30:
			{
				auto auto_diff_energy = fun30(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
				break;
			}
			case 60:
			{
				auto auto_diff_energy = fun60(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
				break;
			}
			case 81:
			{
				auto auto_diff_energy = fun81(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
				break;
			}
		}

		if(grad.size()<=0)
		{
			if(n_bases*size <= SMALL_N)
			{
				auto auto_diff_energy = funN(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
			}
			else if(n_bases * size <= BIG_N)
			{
				auto auto_diff_energy = funBigN(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
			}
			else
			{
				static bool show_message = true;

				if(show_message)
				{
					logger().debug("[Warning] grad {}^{} not using static sizes", n_bases, size);
					show_message = false;
				}

				auto auto_diff_energy = funn(vals, displacement, da);
				grad = auto_diff_energy.getGradient();
			}
		}

		return grad;
	}


	Eigen::MatrixXd hessian_from_energy(const int size, const int n_bases, const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da,
		const std::function<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>		(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun6,
		const std::function<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>		(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun8,
		const std::function<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun12,
		const std::function<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun18,
		const std::function<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun24,
		const std::function<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun30,
		const std::function<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun60,
		const std::function<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun81,
		const std::function<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funN,
		const std::function<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>								(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funn
		)
	{
		Eigen::MatrixXd hessian;

		switch(size * n_bases)
		{
			case 6:
			{
				auto auto_diff_energy = fun6(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
				break;
			}
			case 8:
			{
				auto auto_diff_energy = fun8(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
				break;
			}
			case 12:
			{
				auto auto_diff_energy = fun12(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
				break;
			}
			case 18:
			{
				auto auto_diff_energy = fun18(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
				break;
			}
			case 24:
			{
				auto auto_diff_energy = fun24(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
				break;
			}
			case 30:
			{
				auto auto_diff_energy = fun30(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
				break;
			}
			case 60:
			{
				auto auto_diff_energy = fun60(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
				break;
			}
			case 81:
			{
				auto auto_diff_energy = fun81(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
				break;
			}
		}

		if(hessian.size() <= 0)
		{
// #ifndef POLYFEM_ON_HPC
			// Somehow causes a segfault on the HPC (objects too big for the stack?)
			if(n_bases*size <= SMALL_N)
			{
				auto auto_diff_energy = funN(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
			}
			else
// #endif
			{
				static bool show_message = true;

				if(show_message)
				{
					logger().debug("[Warning] hessian {}*{} not using static sizes", n_bases, size);
					show_message = false;
				}

				auto auto_diff_energy = funn(vals, displacement, da);
				hessian = auto_diff_energy.getHessian();
			}
		}

		// time.stop();
		// std::cout << "-- hessian: " << time.getElapsedTime() << std::endl;

		return hessian;
	}


	double convert_to_lambda(const bool is_volume, const double E, const double nu)
	{
		const double lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));

		if(is_volume)
			return lam;

		const double mmu = convert_to_mu(E, nu);
		return (2.0 * mmu * lam) / (lam + 2 * mmu);
	}

	double convert_to_mu(const double E, const double nu)
	{
		return E / (2.0 * (1.0 + nu));
	}


	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress)
	{
		double von_mises_stress = 0.5 * ( stress(0, 0) - stress(1, 1) ) * ( stress(0, 0) - stress(1, 1) ) + 3.0  *  stress(0, 1) * stress(1, 0);

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
			stifness_tensor_.resize(3, 3);
		else
			stifness_tensor_.resize(6, 6);

		stifness_tensor_.setZero();

		size_ = size;
	}

	double ElasticityTensor::operator()(int i, int j) const
	{
		if(j < i)
		{
			std::swap(i, j);
		}

		assert(j>=i);
		return stifness_tensor_(i, j);
	}

	double &ElasticityTensor::operator()(int i, int j)
	{
		if(j < i)
		{
			std::swap(i, j);
		}

		assert(j>=i);
		return stifness_tensor_(i, j);
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

	void ElasticityTensor::set_from_young_poisson(const double young, const double nu)
	{
		if(size_ == 2)
		{
			stifness_tensor_ <<
			1.0, nu, 0.0,
			nu, 1.0, 0.0,
			0.0, 0.0, (1.0 - nu) / 2.0;
			stifness_tensor_ *= young / (1.0 - nu * nu);
		}
		else
		{
			assert(size_ == 3);
			const double v = nu;
			stifness_tensor_ <<
			1. - v, v, v, 0, 0, 0,
			v, 1. - v, v, 0, 0, 0,
			v, v, 1. - v, 0, 0, 0,
			0, 0, 0, (1. - 2.*v)/2., 0, 0,
			0, 0, 0, 0, (1. - 2.*v)/2., 0,
			0, 0, 0, 0, 0, (1. - 2.*v)/2.;
			stifness_tensor_ *= young / ((1. + v) * (1. - 2. * v));
		}
	}

	void ElasticityTensor::set_orthotropic(
		double   Ex, double   Ey, double   Ez,
		double nuYX, double nuZX, double nuZY,
		double muYZ, double muZX, double muXY)
	{
		//copied from Julian
		assert(size_ == 3);
        // Note: this isn't the flattened compliance tensor! Rather, it is the
        // matrix inverse of the flattened elasticity tensor. See the tensor
        // flattening writeup.
		stifness_tensor_ <<
		1.0 / Ex, -nuYX / Ey, -nuZX / Ez,   0.0,        0.0,        0.0,
		0.0,  	 1.0 / Ey, -nuZY / Ez,      0.0,        0.0,        0.0,
		0.0,        0.0,   1.0 / Ez,        0.0,        0.0,        0.0,
		0.0,        0.0,        0.0, 1.0 / muYZ,        0.0,        0.0,
		0.0,        0.0,        0.0,        0.0, 1.0 / muZX,        0.0,
		0.0,        0.0,        0.0,        0.0,        0.0, 1.0 / muXY;
	}

	void ElasticityTensor::set_orthotropic(double Ex, double Ey, double nuYX, double muXY)
	{
		//copied from Julian
		assert(size_ == 2);
        // Note: this isn't the flattened compliance tensor! Rather, it is the
        // matrix inverse of the flattened elasticity tensor.
		stifness_tensor_ <<
		1.0 / Ex, -nuYX / Ey,   0.0,
		0.0,   1.0 / Ey,        0.0,
		0.0,        0.0, 		1.0 / muXY;
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
