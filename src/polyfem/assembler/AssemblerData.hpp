#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>

namespace polyfem::assembler
{
	class NonLinearAssemblerData
	{
	public:
		NonLinearAssemblerData(
			const ElementAssemblyValues &vals,
			const double dt,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const QuadratureVector &da)
			: vals(vals), dt(dt), x(x), x_prev(x_prev), da(da)
		{
		}

		const ElementAssemblyValues &vals;
		const double dt;
		const Eigen::MatrixXd &x;
		const Eigen::MatrixXd &x_prev;
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
