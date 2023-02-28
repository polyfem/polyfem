#include "OgdenElasticity.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

namespace polyfem::assembler
{
	namespace
	{
		void set_from_json(const json &j, Eigen::VectorXd &v)
		{
			if (j.is_array())
				v = j.get<Eigen::VectorXd>();
			else
				v.setConstant(1, j.get<double>());
		}
	} // namespace

	UnconstrainedOgdenElasticity::UnconstrainedOgdenElasticity()
	{
		alphas_.setOnes(1);
		mus_.setOnes(1);
		Ds_.setOnes(1);
	}

	void UnconstrainedOgdenElasticity::add_multimaterial(const int index, const json &params)
	{
		if (params.contains("alphas"))
			set_from_json(params["alphas"], alphas_);
		if (params.contains("mus"))
			set_from_json(params["mus"], mus_);
		if (params.contains("mus"))
			set_from_json(params["Ds"], Ds_);

		assert(alphas_.size() == mus_.size());
		assert(alphas_.size() == Ds_.size());
	}

	template <typename T>
	T UnconstrainedOgdenElasticity::elastic_energy_T(
		const RowVectorNd &p,
		const int el_id,
		const DefGradMatrix<T> &def_grad) const
	{

		Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> eigs;

		if (size() == 2)
			autogen::eigs_2d<T>(def_grad, eigs);
		else if (size() == 3)
			autogen::eigs_3d<T>(def_grad, eigs);
		else
			assert(false);

		const T J = utils::determinant(def_grad);
		const T Jdenom = pow(J, -1. / size());

		for (long i = 0; i < eigs.size(); ++i)
			eigs(i) = eigs(i) * Jdenom;

		auto val = T(0);
		for (long N = 0; N < alphas_.size(); ++N)
		{
			auto tmp = T(-size());
			const double alpha = alphas_(N);
			const double mu = mus_(N);

			for (long i = 0; i < eigs.size(); ++i)
				tmp += pow(eigs(i), alpha);

			val += 2 * mu / (alpha * alpha) * tmp;
		}

		for (long N = 0; N < Ds_.size(); ++N)
		{
			const double D = Ds_(N);

			val += 1. / D * pow(J - T(1), 2 * (N + 1));
		}

		return val;
	}

	// =========================================================================

	IncompressibleOgdenElasticity::IncompressibleOgdenElasticity()
	{
		coefficients_.setOnes(1);
		expoenents_.setOnes(1);
		bulk_modulus_ = 1.0;
	}

	void IncompressibleOgdenElasticity::add_multimaterial(const int index, const json &params)
	{
		if (params.contains("c"))
			set_from_json(params["c"], coefficients_);
		if (params.contains("m"))
			set_from_json(params["m"], expoenents_);
		if (params.contains("k"))
			bulk_modulus_ = params["k"];

		assert(coefficients_.size() == expoenents_.size());
	}

	template <typename T>
	T IncompressibleOgdenElasticity::elastic_energy_T(
		const RowVectorNd &p,
		const int el_id,
		const DefGradMatrix<T> &def_grad) const
	{
		Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> eigs;

		const T J = polyfem::utils::determinant(def_grad);
		const T log_J = log(J);

		const auto F_tilde = def_grad / pow(J, 1 / 3.0);
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
			const double c = coefficients_[i];
			const double m = expoenents_[i];

			auto tmp = T(-size());
			for (long j = 0; j < eigs.size(); ++j)
				tmp += pow(eigs(j), m);

			val += c / (m * m) * tmp;
		}
		val += 0.5 * bulk_modulus() * log_J * log_J;

		return val;
	}

	// =========================================================================

	// This macro defines the template specializations for UnconstrainedOgdenElasticity::elastic_energy_T
	POLYFEM_TEMPLATE_SPECIALIZE_ELASTIC_ENERGY(UnconstrainedOgdenElasticity)
	// This macro defines the template specializations for IncompressibleOgdenElasticity::elastic_energy_T
	POLYFEM_TEMPLATE_SPECIALIZE_ELASTIC_ENERGY(IncompressibleOgdenElasticity)
} // namespace polyfem::assembler
