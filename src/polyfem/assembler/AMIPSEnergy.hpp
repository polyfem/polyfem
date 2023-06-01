#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
#include <iostream>
#include <vector>

namespace polyfem::assembler
{
	class AMIPSEnergy : public GenericElastic
	{
	public:
		AMIPSEnergy();

		// sets material params
		void add_multimaterial(const int index, const json &params) override;

		std::string name() const override { return "AMIPS"; }
		std::map<std::string, ParamFunc> parameters() const override;

		POLYFEM_OVERRIDE_ELASTIC_ENERGY

	private:
		template <typename T>
		T elastic_energy_T(
			const RowVectorNd &p,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const;

		std::vector<Eigen::MatrixXd> canonical_transformation_;
	};
} // namespace polyfem::assembler