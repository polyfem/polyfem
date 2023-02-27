#include "OgdenElasticity.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

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
			alphas_ = params["alphas"].get<Eigen::VectorXd>();
		}
		if (params.count("mus"))
		{
			mus_ = params["mus"].get<Eigen::VectorXd>();
		}
		if (params.count("Ds"))
		{
			Ds_ = params["Ds"].get<Eigen::VectorXd>();
		}

		assert(alphas_.size() == mus_.size());
		assert(Ds_.size() == mus_.size());
	}

} // namespace polyfem::assembler
