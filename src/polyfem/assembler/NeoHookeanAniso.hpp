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
			const auto right_cauchy_green = def_grad * def_grad.transpose();
			const auto powJ = pow(J, -2. / 3);
			const auto TrB = right_cauchy_green.trace();
			const auto I1_tilde = powJ * (TrB + (3 - size())) - 3;

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> A1;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> A2;

			const T I4_tilde = powJ * (right_cauchy_green.array() * A1.array()).sum();
			const T I6_tilde = powJ * (right_cauchy_green.array() * A2.array()).sum();

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
	};
} // namespace polyfem::assembler