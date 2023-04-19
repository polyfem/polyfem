#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

namespace polyfem::assembler
{
	class UnconstrainedOgdenElasticity : public GenericElastic
	{
	public:
		UnconstrainedOgdenElasticity();

		// sets material params
		void add_multimaterial(const int index, const json &params) override;

		const GenericMatParams &alphas() const { return alphas_; }
		const GenericMatParams &mus() const { return mus_; }
		const GenericMatParams &Ds() const { return Ds_; }

		std::string name() const override { return "UnconstrainedOgden"; }
		std::map<std::string, ParamFunc> parameters() const override;

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

		GenericMatParams alphas_;
		GenericMatParams mus_;
		GenericMatParams Ds_;
	};

	class IncompressibleOgdenElasticity : public GenericElastic
	{
	public:
		IncompressibleOgdenElasticity();

		// sets material params
		void add_multimaterial(const int index, const json &params) override;

		/// Coefficient of nth term, where n can range from 1 to 6
		const GenericMatParams &coefficients() const { return coefficients_; }
		/// Exponent of nth term, where n can range from 1 to 6
		const GenericMatParams &expoenents() const { return expoenents_; }
		/// Bulk modulus
		const GenericMatParam &bulk_modulus() const { return bulk_modulus_; }

		/// Number of terms in the Ogden model
		int num_terms() const { return coefficients_.size(); }

		std::string name() const override { return "IncompressibleOgden"; }
		std::map<std::string, ParamFunc> parameters() const override;

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

		GenericMatParams coefficients_;
		GenericMatParams expoenents_;
		GenericMatParam bulk_modulus_;
	};
} // namespace polyfem::assembler
