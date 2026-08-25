#pragma once

#include <polyfem/assembler/GenericFiber.hpp>
#include <polyfem/assembler/GenericElastic.hpp>

namespace polyfem::assembler
{
	// Modified GOH anisotropic fiber model: adds fiber dispersion (kappa) and a
	// smooth (C-infinity) distension/compression switch, built on the FULL
	// invariants I1 = tr(C) and I4 = a0.C a0. It is intentionally kept separate
	// from HGOFiber (the classic aligned HGO-2000 decoupled form, isochoric I4,
	// hard tension-only cutoff), which is left unchanged. Implemented energy:
	//
	//   psi = (k1 / (2 k2)) * X(E4) * ( exp(k2 * E4^2) - 1 )
	//   E4  = kappa * I1 + (1 - d*kappa) * I4 - 1        (d = spatial dimension)
	//   X   = 1 / ( 1 + exp(-k_chi * E4) )               (logistic, centered at E4 = 0)
	//
	// Parameter naming: k1 == manuscript a_f, k2 == manuscript b_f (names kept to
	// match HGOFiber and existing inputs).
	class HGODispersion : public GenericFiber<HGODispersion>
	{
	public:
		HGODispersion();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		std::string name() const override { return "HGODispersion"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			const double k1 = k1_(p, t, el_id);       // fiber stiffness (manuscript a_f)
			const double k2 = k2_(p, t, el_id);       // exp. stiffening (manuscript b_f)
			const double kappa = kappa_(p, t, el_id); // dispersion in [0, 1/d]; absent => 0

			// Modified-anisotropy GOH invariant from the FULL invariants:
			//   E4 = kappa * I1 + (1 - d*kappa) * I4 - 1
			// The (1 - d*kappa) factor keeps the generalized structure tensor
			// unit-trace; at kappa = 0 this collapses to I4 - 1 (aligned limit).
			const double d = static_cast<double>(this->size());
			const T i1 = I1(def_grad);
			const T i4 = I4(p, t, el_id, def_grad);
			const T E4 = kappa * i1 + (1.0 - d * kappa) * i4 - 1.0;

			// Smooth logistic distension/compression switch, centered at E4 = 0
			// (unloaded state I1 = d, I4 = 1 => E4 = 0). Replaces the C0 hard
			// tension-only cutoff so the energy (and its autodiff gradient and
			// Hessian) is C-infinity.
			const T chi = 1.0 / (1.0 + exp(-k_chi_ * E4));

			return (k1 / (2.0 * k2)) * chi * (exp(k2 * E4 * E4) - 1.0);
		}

	private:
		// Full first invariant I1 = tr(C) = tr(F^T F) = sum_ij F_ij^2 (NOT isochoric).
		template <typename T>
		T I1(const DefGradMatrix<T> &def_grad) const
		{
			T res = T(0);
			for (int i = 0; i < def_grad.rows(); ++i)
				for (int j = 0; j < def_grad.cols(); ++j)
					res += def_grad(i, j) * def_grad(i, j);
			return res;
		}

		// Full fourth invariant I4 = a0 . C a0 with a0 normalized to unit length
		// (normalize = true) and the FULL C (isocoric = false, no J^{-2/3}).
		// Reuses the tested GenericFiber::I4Bar_generic machinery; normalizing
		// makes the term independent of the input fiber vector's magnitude.
		template <typename T>
		T I4(const RowVectorNd &p,
			 const double t,
			 const int el_id,
			 const DefGradMatrix<T> &def_grad) const
		{
			return this->I4Bar_generic(p, t, el_id, def_grad, /*normalize=*/true, /*isocoric=*/false);
		}

		GenericMatParam k1_;    // fiber stiffness  (manuscript a_f)
		GenericMatParam k2_;    // exp. stiffening  (manuscript b_f)
		GenericMatParam kappa_; // fiber dispersion kappa (in [0, 1/d]); absent => 0
		double k_chi_ = 100.0;  // logistic smoothness k_X; manuscript fixes 100
	};
} // namespace polyfem::assembler
