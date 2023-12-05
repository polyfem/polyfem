#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>

namespace polyfem::assembler
{
	class NonLinearAssemblerData
	{
	public:
		NonLinearAssemblerData(
			const ElementAssemblyValues &vals,
			const double t,
			const double dt,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const QuadratureVector &da)
			: vals(vals), t(t), dt(dt), x(x), x_prev(x_prev), da(da)
		{
		}

		const ElementAssemblyValues &vals;
		const double t;
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
			const double t,
			int i, int j,
			const QuadratureVector &da)
			: vals(vals), t(t), i(i), j(j), da(da)
		{
		}

		/// stores the evaluation for that element
		const ElementAssemblyValues &vals;

		const double t;
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
			const double t,
			int i, int j,
			const QuadratureVector &da)
			: psi_vals(psi_vals), phi_vals(phi_vals),
			  t(t), i(i), j(j), da(da)
		{
		}

		/// stores the evaluation for that element
		const ElementAssemblyValues &psi_vals;
		/// stores the evaluation for that element
		const ElementAssemblyValues &phi_vals;

		const double t;
		/// first local order
		const int i;
		/// second local order
		const int j;
		/// contains both the quadrature weight and the change of metric in the integral
		const QuadratureVector &da;
	};

	class OptAssemblerData
	{
	public:
		OptAssemblerData(
			const double t,
			const double dt,
			const int el_id,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &global_pts,
			const Eigen::MatrixXd &grad_u_i)
			: t(t), dt(dt), el_id(el_id), local_pts(local_pts), global_pts(global_pts), grad_u_i(grad_u_i)
		{
		}

		const double t;
		const double dt;
		const int el_id;
		const Eigen::MatrixXd &local_pts;
		const Eigen::MatrixXd &global_pts;
		const Eigen::MatrixXd &grad_u_i;
	};

	class OutputData
	{
	public:
		OutputData(
			const double t,
			const int el_id,
			const basis::ElementBases &bs,
			const basis::ElementBases &gbs,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun)
			: t(t), el_id(el_id), bs(bs), gbs(gbs), local_pts(local_pts), fun(fun)
		{
		}

		const double t;
		const int el_id;
		const basis::ElementBases &bs;
		const basis::ElementBases &gbs;
		const Eigen::MatrixXd &local_pts;
		const Eigen::MatrixXd &fun;
	};
} // namespace polyfem::assembler
