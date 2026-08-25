#pragma once

#include <polyfem/assembler/GenericFiber.hpp>
#include <polyfem/assembler/GenericElastic.hpp>

#include <map>
#include <string>

namespace polyfem::assembler
{
	class ActiveFiber : public GenericFiber<ActiveFiber>
	{
	public:
		ActiveFiber();

		// JSON params:
		//  - "Tmax": scalar / field / param expression (GenericMatParam)
		//  - "activation": scalar / field / param expression in [0,1] (GenericMatParam)
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		std::string name() const override { return "ActiveFiber"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			// activation scalar (your C(t) equivalent)
			const double Tmax = Tmax_(p, t, el_id);
			const double Cact = activation_(p, t, el_id); // should be in [0,1]

			// // FEBio constants
			// const double Ca0 = 4.35;
			// const double Ca0max = 4.35;
			// const double B = 4.75;
			// const double l0 = 1.58;
			// const double lr = 2.04;

			// // invariant (isochoric like HGO)
			// const T I4bar = I4Bar(p, t, el_id, def_grad);
			const T I4bar = I4Bar_with_norm(p, t, el_id, def_grad);
			// const T lambda = sqrt(I4bar); // if you can compute non-isochoric I4, use that here instead
			// const T l = lambda * T(lr);

			// // No active tension if l <= l0
			// if (l <= T(l0))
			// 	return T(0);

			// // ECa50 and Ca-sensitivity factor
			// const T denom = exp(T(B) * (l - T(l0))) - T(1);
			// const T ECa50 = T(Ca0max) / sqrt(denom);
			// const T cst = T(Ca0 * Ca0) / (T(Ca0 * Ca0) + ECa50 * ECa50);

			// // Total scalar active tension multiplier
			// const T Ta = T(Tmax) * T(Cact) * cst;

			const double a = std::min(1.0, std::max(0.0, Cact));
			const T Ta = T(Tmax * a);

			// Energy
			return T(-0.5) * Ta * (I4bar - T(1));
		}

	private:
		GenericMatParam Tmax_;
		GenericMatParam activation_;
	};
} // namespace polyfem::assembler