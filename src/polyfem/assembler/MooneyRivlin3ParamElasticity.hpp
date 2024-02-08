#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

// non linear MooneyRivlin material model
namespace polyfem::assembler
{
	class MooneyRivlin3ParamElasticity : public GenericElastic<MooneyRivlin3ParamElasticity>
	{
	public:
		MooneyRivlin3ParamElasticity();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		const GenericMatParam &c1() const { return c1_; }
		const GenericMatParam &c2() const { return c2_; }
		const GenericMatParam &c3() const { return c3_; }
		const GenericMatParam &d1() const { return d1_; }

		std::string name() const override { return "MooneyRivlin3Param"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			const double c1 = c1_(p, t, el_id);
			const double c2 = c2_(p, t, el_id);
			const double c3 = c3_(p, t, el_id);
			const double d1 = d1_(p, t, el_id);

			const T J = polyfem::utils::determinant(def_grad);
			const auto right_cauchy_green = def_grad * def_grad.transpose();

			const auto powJ = pow(J, -2. / 3);
			const auto TrB = right_cauchy_green.trace();
			const auto I1_tilde = powJ * (TrB + (3 - size())) - 3;
			const auto second_invariant_val = (size() == 3) ? 0.5 * (TrB * TrB - (right_cauchy_green * right_cauchy_green).trace()) : TrB + J * J;
			const auto I2_tilde = (powJ * powJ) * second_invariant_val - 3;

			const T val = c1 * I1_tilde + (c2 + c3 * I1_tilde) * I2_tilde + d1 * (J - 1) * (J - 1);

			return val;
		}

	private:
		GenericMatParam c1_;
		GenericMatParam c2_;
		GenericMatParam c3_;
		GenericMatParam d1_;
	};
} // namespace polyfem::assembler