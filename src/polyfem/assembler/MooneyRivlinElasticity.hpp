#pragma once

#include "MatParams.hpp"
#include "GenericElastic.hpp"
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
#include <iostream>
#include <vector>

// non linear MooneyRivlin material model
namespace polyfem::assembler
{
	class MooneyRivlinElasticity : public GenericElastic
	{
	public:
		MooneyRivlinElasticity();

		// sets material params
		void add_multimaterial(const int index, const json &params) override;

		const GenericMatParam &c1() const { return c1_; }
		const GenericMatParam &c2() const { return c2_; }
		const GenericMatParam &k() const { return k_; }

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

		GenericMatParam c1_;
		GenericMatParam c2_;
		GenericMatParam k_;
	};
} // namespace polyfem::assembler