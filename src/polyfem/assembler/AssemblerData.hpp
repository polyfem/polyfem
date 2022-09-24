#pragma once

#include "ElementAssemblyValues.hpp"

namespace polyfem::assembler
{
	class NonLinearAssemblerData
	{
	public:
		NonLinearAssemblerData(
			const ElementAssemblyValues &vals,
			const Eigen::MatrixXd &x,
			const QuadratureVector &da)
			: vals(vals), x(x), da(da)
		{
		}

		const ElementAssemblyValues &vals;
		const Eigen::MatrixXd &x;
		const QuadratureVector &da;
	};

	class LinearAssemblerData
	{
	public:
		LinearAssemblerData(
			const ElementAssemblyValues &vals,
			int i, int j,
			const QuadratureVector &da)
			: vals(vals), i(i), j(j), da(da)
		{
		}

		/// stores the evaluation for that element
		const ElementAssemblyValues &vals;
		/// first local order
		const int i;
		/// second local order
		const int j;
		/// contains both the quadrature weight and the change of metric in the integral
		const QuadratureVector &da;
	};

	class MixedAssemblerData
	{
	public:
		MixedAssemblerData(
			const ElementAssemblyValues &psi_vals,
			const ElementAssemblyValues &phi_vals,
			int i, int j,
			const QuadratureVector &da)
			: psi_vals(psi_vals), phi_vals(phi_vals),
			  i(i), j(j), da(da)
		{
		}

		/// stores the evaluation for that element
		const ElementAssemblyValues &psi_vals;
		/// stores the evaluation for that element
		const ElementAssemblyValues &phi_vals;
		/// first local order
		const int i;
		/// second local order
		const int j;
		/// contains both the quadrature weight and the change of metric in the integral
		const QuadratureVector &da;
	};
} // namespace polyfem::assembler
