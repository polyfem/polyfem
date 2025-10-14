#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

namespace polyfem::assembler
{
	class IsochoricNeoHookean : public GenericElastic<IsochoricNeoHookean>
	{
	public:
		IsochoricNeoHookean();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		std::string name() const override { return "IsochoricNeoHookean"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			double lambda, mu;
			params_.lambda_mu(p, p, t, el_id, lambda, mu);

			const T J = polyfem::utils::determinant(def_grad);
			const auto Cbar = (def_grad.transpose() * def_grad / pow(J, 2.0 / 3.0)).eval();
			const T I1_bar = Cbar.trace();

			// Isochoric NK strain energy: Psi_iso = mu/2 * (I1_bar - 3)
			return (mu / 2.0) * (I1_bar - T(3.0));
		}

	private:
		LameParameters params_;
	};
} // namespace polyfem::assembler