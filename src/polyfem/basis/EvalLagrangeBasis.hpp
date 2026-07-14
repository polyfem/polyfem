#pragma once

#include <polyfem/basis/BasisStore.hpp>

#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>

namespace polyfem::basis
{

	/// @brief Compute lagrange basis count in element.
	POLYFEM_BOTH int lagrange_basis_count(const BasisDesc &desc);

	/// @brief Compute lagrange basis value for one local node.
	///
	/// Expected output size: `x.size()`.
	/// Layout: [ ϕ(q0) ϕ(q1) ... ]
	///
	/// @param local_basis_index Element local index for basis.
	/// @param desc Basis descriptor.
	/// @param store Basis store view.
	/// @param x Quadrature point component x.
	/// @param y Quadrature point component y. Might be empty in 1D.
	/// @param z Quadrature point component z. Might be empty in 1D/2D.
	/// @param values Output basis value.
	POLYFEM_BOTH void lagrange_basis_values_single(
		int local_basis_index,
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values);

	/// @brief Compute lagrange basis value.
	///
	/// Expected output size: `lagrange_basis_count(desc) * x.size()`.
	/// Layout: [ ϕ0(q0) ϕ0(q1) ... ] [ ϕ1(q0) ϕ1(q1) ... ] ...
	///
	/// @param desc Basis descriptor.
	/// @param store Basis store view.
	/// @param x Quadrature point component x.
	/// @param y Quadrature point component y. Might be empty in 1D.
	/// @param z Quadrature point component z. Might be empty in 1D/2D.
	/// @param values Output basis value.
	POLYFEM_BOTH void lagrange_basis_values(
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values);

	/// @brief Compute lagrange basis gradient w.r.t parametric space (i.e. barycentric coordinate) for a single local node.
	///
	/// Expected output size for each non-empty gradient component: `x.size()`.
	/// Layout: [ ∂ϕ(q0) ∂ϕ(q1) ... ]
	///
	/// @param local_basis_index Element local index for basis.
	/// @param desc Basis descriptor.
	/// @param store Basis store view.
	/// @param x Quadrature point component x.
	/// @param y Quadrature point component y. Might be empty in 1D.
	/// @param z Quadrature point component z. Might be empty in 1D/2D.
	/// @param grad_x Output quadrature grad component x.
	/// @param grad_y Output quadrature grad component y. Might be empty in 1D.
	/// @param grad_z Output quadrature grad component z. Might be empty in 1D/2D.
	POLYFEM_BOTH void lagrange_basis_gradients_single(
		int local_basis_index,
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z);

	/// @brief Compute lagrange basis gradient w.r.t parametric space (i.e. barycentric coordinate)
	///
	/// Expected output size for each non-empty gradient component:
	/// `lagrange_basis_count(desc) * x.size()`.
	/// Layout inside each component: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
	///
	/// @param desc Basis descriptor.
	/// @param store Basis store view.
	/// @param x Quadrature point component x.
	/// @param y Quadrature point component y. Might be empty in 1D.
	/// @param z Quadrature point component z. Might be empty in 1D/2D.
	/// @param grad_x Output quadrature grad component x.
	/// @param grad_y Output quadrature grad component y. Might be empty in 1D.
	/// @param grad_z Output quadrature grad component z. Might be empty in 1D/2D.
	POLYFEM_BOTH void lagrange_basis_gradients(
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z);

	/// @brief Compute lagrange basis values and gradients.
	///
	/// Expected output size for `values` and each non-empty gradient component:
	/// `lagrange_basis_count(desc) * x.size()`.
	/// Layout: [ ϕ0(q0) ϕ0(q1) ... ] [ ϕ1(q0) ϕ1(q1) ... ] ...
	/// Gradient layout inside each component: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
	///
	/// @param desc Basis descriptor.
	/// @param store Basis store view.
	/// @param x Quadrature point component x.
	/// @param y Quadrature point component y. Might be empty in 1D.
	/// @param z Quadrature point component z. Might be empty in 1D/2D.
	/// @param values Output basis value.
	/// @param grad_x Output quadrature grad component x.
	/// @param grad_y Output quadrature grad component y. Might be empty in 1D.
	/// @param grad_z Output quadrature grad component z. Might be empty in 1D/2D.
	POLYFEM_BOTH void lagrange_basis_value_and_gradients(
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z);

} // namespace polyfem::basis
