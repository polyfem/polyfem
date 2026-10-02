#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>

#include <functional>
#include <vector>

namespace polyfem::assembler
{
	/// Local data shared by assemblers involving an arbitrary number of FE spaces.
	/// The data is a non-owning view; element traversal and global gather/scatter are
	/// deliberately handled by the form using the assembler.
	class MultiSpacesNLAssemblerData
	{
	public:
		using Values = std::vector<std::reference_wrapper<const ElementAssemblyValues>>;
		using Coefficients = std::vector<std::reference_wrapper<const Eigen::VectorXd>>;

		MultiSpacesNLAssemblerData(
			Values vals,
			Coefficients x,
			Coefficients x_prev,
			const double t,
			const double dt,
			const QuadratureVector &da)
			: vals_(std::move(vals)),
			  x_(std::move(x)),
			  x_prev_(std::move(x_prev)),
			  t(t),
			  dt(dt),
			  da(da)
		{
			assert(vals_.size() == x_.size());
			assert(x_prev_.empty() || vals_.size() == x_prev_.size());
		}

		virtual ~MultiSpacesNLAssemblerData() = default;

		int n_spaces() const { return int(vals_.size()); }
		const ElementAssemblyValues &vals(const int space) const { return vals_.at(space).get(); }
		const Eigen::VectorXd &x(const int space) const { return x_.at(space).get(); }
		const Eigen::VectorXd &x_prev(const int space) const { return x_prev_.at(space).get(); }

		const double t;
		const double dt;
		const QuadratureVector &da;

	private:
		Values vals_;
		Coefficients x_;
		Coefficients x_prev_;
	};

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

	class MixedNonLinearAssemblerData
	{
	public:
		MixedNonLinearAssemblerData(
			const ElementAssemblyValues &psi_vals,
			const ElementAssemblyValues &phi_vals,
			const double t,
			const double dt,
			const Eigen::MatrixXd &x_phi,
			const Eigen::MatrixXd &x_psi,
			const Eigen::MatrixXd &x_phi_prev,
			const Eigen::MatrixXd &x_psi_prev,
			const QuadratureVector &da)
			: psi_vals(psi_vals), phi_vals(phi_vals),
			  t(t), dt(dt),
			  x_phi(x_phi), x_psi(x_psi),
			  x_phi_prev(x_phi_prev), x_psi_prev(x_psi_prev),
			  da(da)
		{
		}

		/// Values for the second block, historically scalar pressure-like bases.
		const ElementAssemblyValues &psi_vals;
		/// Values for the first block, historically tensor velocity/displacement-like bases.
		const ElementAssemblyValues &phi_vals;

		const double t;
		const double dt;
		const Eigen::MatrixXd &x_phi;
		const Eigen::MatrixXd &x_psi;
		const Eigen::MatrixXd &x_phi_prev;
		const Eigen::MatrixXd &x_psi_prev;
		/// Contains both the quadrature weight and the change of metric in the integral.
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
