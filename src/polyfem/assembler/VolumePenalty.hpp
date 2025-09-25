#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

namespace polyfem::assembler
{
	class VolumePenalty : public GenericElastic<VolumePenalty>
	{
	public:
		VolumePenalty();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		std::string name() const override { return "VolumePenalty"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			const double k = k_(p, t, el_id);

			const T J = polyfem::utils::determinant(def_grad);
			const T log_J = log(J);

			const T val = k / 2.0 * ((J * J - T(1)) / 2.0 - log_J);

			return val;
		}

	private:
		GenericMatParam k_;
	};
} // namespace polyfem::assembler