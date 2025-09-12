#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

// non linear MooneyRivlin material model
namespace polyfem::assembler
{
	class NeoHookeanAniso : public GenericElastic<NeoHookeanAniso>
	{
	public:
		NeoHookeanAniso();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		const GenericMatParam &c() const { return c_; }
		const GenericMatParam &k1() const { return k1_; }
		const GenericMatParam &k2() const { return k2_; }
		const GenericMatParam &d1() const { return d1_; }
		const DirectionVector &a1() const { return a1_; }
		const DirectionVector &a2() const { return a2_; }

		std::string name() const override { return "NeoHookeanAnisotropic"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			const double c = c_(p, t, el_id);
			const double k1 = k1_(p, t, el_id);
			const double k2 = k2_(p, t, el_id);
			const double d1 = d1_(p, t, el_id);

			const T J = polyfem::utils::determinant(def_grad);
			const T logJ = log(J);
			const auto right_cauchy_green = (def_grad * def_grad.transpose()).eval();
			const T powJ = pow(J, -2. / 3);
			const T TrB = right_cauchy_green.trace();
			const T I1_tilde = powJ * (TrB + (3 - size())) - 3;

			const auto a1 = a1_(p, t, el_id);
			const auto a2 = a2_(p, t, el_id);

			const auto A1 = (a1 * a1.transpose()).eval();
			const auto A2 = (a2 * a2.transpose()).eval();

			T I4_tilde = T(0);
			T I6_tilde = T(0);
			for (int i = 0; i < size(); ++i)
				for (int j = 0; j < size(); ++j)
				{
					I4_tilde += powJ * right_cauchy_green(i, j) * A1(i, j);
					I6_tilde += powJ * right_cauchy_green(i, j) * A2(i, j);
				}

			const T psi_iso = c * I1_tilde;
			const T psi_aniso = (k1 / 2 / k2) * ((exp(k2 * pow(I4_tilde - 1, 2)) - 1) + (exp(k2 * pow(I6_tilde - 1, 2)) - 1));

			const T val = psi_iso + psi_aniso + d1 * logJ * logJ;
			return val;
		}

	private:
		GenericMatParam c_;
		GenericMatParam k1_;
		GenericMatParam k2_;
		GenericMatParam d1_;
		DirectionVector a1_;
		DirectionVector a2_;
	};
} // namespace polyfem::assembler