#pragma once

#include <polyfem/assembler/GenericFiber.hpp>
#include <polyfem/assembler/GenericElastic.hpp>

namespace polyfem::assembler
{
	class HGOFiber : public GenericFiber<HGOFiber>
	{
	public:
		HGOFiber();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		std::string name() const override { return "HGOFiber"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			const double k1 = k1_(p, t, el_id);
			const double k2 = k2_(p, t, el_id);
			const T i4Bar = I4Bar(p, t, el_id, def_grad);

			const T temp = i4Bar - T(1.0);

			return (k1 / (2.0 * k2)) * (exp(k2 * temp * temp) - 1.0);
		}

	private:
		GenericMatParam k1_;
		GenericMatParam k2_;
	};
} // namespace polyfem::assembler