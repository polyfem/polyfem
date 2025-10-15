#include "OgdenElasticity.hpp"

#include <polyfem/autogen/auto_eigs.hpp>

namespace polyfem::assembler
{
	UnconstrainedOgdenElasticity::UnconstrainedOgdenElasticity()
		: alphas_("alphas"), mus_("mus"), Ds_("Ds")
	{
	}

	void UnconstrainedOgdenElasticity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		// TODO check me
		alphas_.add_multimaterial(index, params, "");
		mus_.add_multimaterial(index, params, units.stress());
		Ds_.add_multimaterial(index, params, units.stress());
		assert(alphas_.size() == mus_.size());
		assert(alphas_.size() == Ds_.size());
	}

	std::map<std::string, Assembler::ParamFunc> UnconstrainedOgdenElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &alphas = this->alphas();
		const auto &mus = this->mus();
		const auto &Ds = this->Ds();

		for (int i = 0; i < alphas.size(); ++i)
			res[fmt::format("alpha_{}", i)] = [&alphas, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
				return alphas[i](p, t, e);
			};

		for (int i = 0; i < mus.size(); ++i)
			res[fmt::format("mu_{}", i)] = [&mus, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
				return mus[i](p, t, e);
			};

		for (int i = 0; i < Ds.size(); ++i)
			res[fmt::format("D_{}", i)] = [&Ds, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
				return Ds[i](p, t, e);
			};
		return res;
	}

	// =========================================================================

	IncompressibleOgdenElasticity::IncompressibleOgdenElasticity()
		: coefficients_("c"), expoenents_("m"), bulk_modulus_("k")
	{
	}

	void IncompressibleOgdenElasticity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		coefficients_.add_multimaterial(index, params, units.stress());
		expoenents_.add_multimaterial(index, params, "");
		bulk_modulus_.add_multimaterial(index, params, units.stress());
		assert(coefficients_.size() == expoenents_.size());
	}

	std::map<std::string, Assembler::ParamFunc> IncompressibleOgdenElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;

		const auto &coefficients = this->coefficients();
		const auto &expoenents = this->expoenents();
		const auto &k = this->bulk_modulus();

		for (int i = 0; i < coefficients.size(); ++i)
			res[fmt::format("c_{}", i)] = [&coefficients, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
				return coefficients[i](p, t, e);
			};

		for (int i = 0; i < expoenents.size(); ++i)
			res[fmt::format("m_{}", i)] = [&expoenents, i](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
				return expoenents[i](p, t, e);
			};

		res["k"] = [&k](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return k(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler
