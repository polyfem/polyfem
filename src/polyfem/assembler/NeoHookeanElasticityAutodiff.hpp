#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

/// Used for test only
namespace polyfem::assembler
{
	class NeoHookeanAutodiff : public GenericElastic
	{
	public:
		std::string name() const override { return "NeoHookeanAutodiff"; }
		// Used only for testing, no need to output
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

		NeoHookeanAutodiff();

		// sets material params
		void add_multimaterial(const int index, const json &params) override;

		// This macro defines the overriden functions that compute the energy:
		// template <typename T>
		// T elastic_energy(const RowVectorNd &p, const int el_id, const DefGradMatrix<T> &def_grad) const override { elastic_energy_T<T>(p, el_id, def_grad); };
		POLYFEM_OVERRIDE_ELASTIC_ENERGY

	private:
		template <typename T>
		T elastic_energy_T(
			const RowVectorNd &p,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const;

		LameParameters params_;
	};
} // namespace polyfem::assembler