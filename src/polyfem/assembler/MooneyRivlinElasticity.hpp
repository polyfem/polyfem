#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

// non linear MooneyRivlin material model
namespace polyfem::assembler
{
	class MooneyRivlinElasticity : public GenericElastic<MooneyRivlinElasticity>
	{
	public:
		MooneyRivlinElasticity();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		const GenericMatParam &c1() const { return c1_; }
		const GenericMatParam &c2() const { return c2_; }
		const GenericMatParam &k() const { return k_; }

		std::string name() const override { return "MooneyRivlin"; }
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
			const double k = k_(p, t, el_id);

			const T J = polyfem::utils::determinant(def_grad);
			const T log_J = log(J);

			const auto F_tilde = def_grad / pow(J, 1.0 / size());
			const auto C_tilde = F_tilde * F_tilde.transpose();
			const auto I1_tilde = first_invariant(C_tilde);
			const auto I2_tilde = second_invariant(C_tilde);

			const T val = c1 * (I1_tilde - size()) + c2 * (I2_tilde - size()) + k / 2 * (log_J * log_J);

			return val;
		}

	private:
		GenericMatParam c1_;
		GenericMatParam c2_;
		GenericMatParam k_;
	};
} // namespace polyfem::assembler