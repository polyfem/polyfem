#pragma once
#include "OgdenElasticity.hpp"

#include <polyfem/autogen/auto_eigs.hpp>

namespace polyfem::assembler
{
	template <typename T>
	T UnconstrainedOgdenElasticity::elastic_energy(
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

	template <typename T>
	T IncompressibleOgdenElasticity::elastic_energy(
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
} // namespace polyfem::assembler
