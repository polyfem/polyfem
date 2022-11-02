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

	void OgdenElasticity::add_multimaterial(const int index, const json &params, const int size)
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

} // namespace polyfem::assembler
