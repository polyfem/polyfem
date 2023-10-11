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
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			const double t = 0; // TODO

			const double c1 = c1_(p, t, el_id);
			const double c2 = c2_(p, t, el_id);
			const double c3 = c3_(p, t, el_id);
			const double d1 = d1_(p, t, el_id);

			const T J = polyfem::utils::determinant(def_grad);
			const T log_J = log(J);
			const auto F_tilde = def_grad;
			const auto right_cauchy_green = F_tilde * F_tilde.transpose();

			const auto I1_tilde = pow(J, -2. / size()) * first_invariant(right_cauchy_green);
			const auto I2_tilde = pow(J, -4. / size()) * second_invariant(right_cauchy_green);

			const T val = c1 * (I1_tilde - size()) + c2 * (I2_tilde - size()) + c3 * (I1_tilde - size()) * (I2_tilde - size()) + d1 * log_J * log_J;

			return val;
		}

	private:
		GenericMatParam c1_;
		GenericMatParam c2_;
		GenericMatParam c3_;
		GenericMatParam d1_;
	};
} // namespace polyfem::assembler