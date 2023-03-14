#include "OgdenElasticity.hpp"

#include <polyfem/autogen/auto_eigs.hpp>

namespace polyfem::assembler
{
	UnconstrainedOgdenElasticity::UnconstrainedOgdenElasticity()
		: alphas_("alphas"), mus_("mus"), Ds_("Ds")
	{
	}

	void UnconstrainedOgdenElasticity::add_multimaterial(const int index, const json &params)
	{
		alphas_.add_multimaterial(index, params);
		mus_.add_multimaterial(index, params);
		Ds_.add_multimaterial(index, params);
		assert(alphas_.size() == mus_.size());
		assert(alphas_.size() == Ds_.size());
	}

	template <typename T>
	T UnconstrainedOgdenElasticity::elastic_energy_T(
		const RowVectorNd &p,
		const int el_id,
		const DefGradMatrix<T> &def_grad) const
	{
		constexpr double t = 0; // TODO if we want to allow material that varys over time

		Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> eigs;

		if (size() == 2)
		{
			// No need to symmetrize F to compute eigen values analytically
			autogen::eigs_2d<T>(def_grad, eigs);
		}
		else
		{
			assert(size() == 3);
			// Symmetrize F to compute eigen values analytically
			autogen::eigs_3d<T>(def_grad.transpose() * def_grad, eigs);
			eigs = sqrt(eigs.array());
		}

		const T J = utils::determinant(def_grad);
		const T Jdenom = pow(J, -1. / size());

		for (long i = 0; i < eigs.size(); ++i)
			eigs(i) = eigs(i) * Jdenom;

		auto val = T(0);
		for (long N = 0; N < alphas_.size(); ++N)
		{
			auto tmp = T(-size());
			const double alpha = alphas_[N](p, t, el_id);
			const double mu = mus_[N](p, t, el_id);

			for (long i = 0; i < eigs.size(); ++i)
				tmp += pow(eigs(i), alpha);

			val += 2 * mu / (alpha * alpha) * tmp;
		}

		for (long N = 0; N < Ds_.size(); ++N)
		{
			const double D = Ds_[N](p, t, el_id);

			val += 1. / D * pow(J - T(1), 2 * (N + 1));
		}

		return val;
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

	void IncompressibleOgdenElasticity::add_multimaterial(const int index, const json &params)
	{
		coefficients_.add_multimaterial(index, params);
		expoenents_.add_multimaterial(index, params);
		bulk_modulus_.add_multimaterial(index, params);
		assert(coefficients_.size() == expoenents_.size());
	}

	template <typename T>
	T IncompressibleOgdenElasticity::elastic_energy_T(
		const RowVectorNd &p,
		const int el_id,
		const DefGradMatrix<T> &def_grad) const
	{
		constexpr double t = 0; // TODO if we want to allow material that varys over time

		Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> eigs;

		const T J = polyfem::utils::determinant(def_grad);
		const T log_J = log(J);

		const auto F_tilde = def_grad / pow(J, 1.0 / size());
		const auto C_tilde = F_tilde * F_tilde.transpose();

		if (size() == 2)
			autogen::eigs_2d<T>(C_tilde, eigs);
		else if (size() == 3)
			autogen::eigs_3d<T>(C_tilde, eigs);
		else
			assert(false);
		eigs = sqrt(eigs.array());

		T val = T(0);
		for (long i = 0; i < num_terms(); ++i)
		{
			const double c = coefficients_[i](p, t, el_id);
			const double m = expoenents_[i](p, t, el_id);

			auto tmp = T(-size());
			for (long j = 0; j < eigs.size(); ++j)
				tmp += pow(eigs(j), m);

			val += c / (m * m) * tmp;
		}
		val += 0.5 * bulk_modulus_(p, t, el_id) * log_J * log_J;

		return val;
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

	// =========================================================================

	// This macro defines the template specializations for UnconstrainedOgdenElasticity::elastic_energy_T
	POLYFEM_TEMPLATE_SPECIALIZE_ELASTIC_ENERGY(UnconstrainedOgdenElasticity)
	// This macro defines the template specializations for IncompressibleOgdenElasticity::elastic_energy_T
	POLYFEM_TEMPLATE_SPECIALIZE_ELASTIC_ENERGY(IncompressibleOgdenElasticity)
} // namespace polyfem::assembler
