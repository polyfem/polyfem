#include "OgdenElasticity.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	namespace
	{
		void fill_mat_from_vect(const std::vector<double> vals, Eigen::VectorXd &vec)
		{
			vec.resize(vals.size());

			for (size_t i = 0; i < vals.size(); ++i)
				vec(i) = vals[i];
		}
	} // namespace

	OgdenElasticity::OgdenElasticity()
	{
		alphas_.resize(1);
		mus_.resize(1);
		Ds_.resize(1);

		alphas_.setOnes();
		mus_.setOnes();
		Ds_.setOnes();
	}

	void OgdenElasticity::add_multimaterial(const int index, const json &params)
	{
		if (params.count("alphas"))
		{
			const std::vector<double> tmp = params["alphas"];
			fill_mat_from_vect(tmp, alphas_);
		}
		if (params.count("mus"))
		{
			const std::vector<double> tmp = params["mus"];
			fill_mat_from_vect(tmp, mus_);
		}
		if (params.count("Ds"))
		{
			const std::vector<double> tmp = params["Ds"];
			fill_mat_from_vect(tmp, Ds_);
		}

		assert(alphas_.size() == mus_.size());
		assert(Ds_.size() == mus_.size());
	}

	void OgdenElasticity::stress_from_disp_grad(const int size,
												const RowVectorNd &p,
												const int el_id,
												const Eigen::MatrixXd &displacement_grad,
												Eigen::MatrixXd &stress_tensor) const
	{
		stress_tensor.resize(size, size);

		logger().warn("Stress tensor not implementd");

		// 	if(size() == 2)
		// 	{
		// 		std::array<double, 3> eps;
		// 		eps[0] = strain(0,0);
		// 		eps[1] = strain(1,1);
		// 		eps[2] = 2*strain(0,1);

		// 		stress_tensor <<
		// 		stress(eps, 0), stress(eps, 2),
		// 		stress(eps, 2), stress(eps, 1);
		// 	}
		// 	else
		// 	{
		// 		std::array<double, 6> eps;
		// 		eps[0] = strain(0,0);
		// 		eps[1] = strain(1,1);
		// 		eps[2] = strain(2,2);
		// 		eps[3] = 2*strain(1,2);
		// 		eps[4] = 2*strain(0,2);
		// 		eps[5] = 2*strain(0,1);

		// 		stress_tensor <<
		// 		stress(eps, 0), stress(eps, 5), stress(eps, 4),
		// 		stress(eps, 5), stress(eps, 1), stress(eps, 3),
		// 		stress(eps, 4), stress(eps, 3), stress(eps, 2);
		// 	}

		// 	stress_tensor = (Eigen::MatrixXd::Identity(size(), size()) + displacement_grad) * stress_tensor;

		// 	all.row(p) = fun(stress_tensor);
		// }
	}

} // namespace polyfem::assembler
