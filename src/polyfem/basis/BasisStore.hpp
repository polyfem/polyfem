#pragma once

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>
#include <functional>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/CudaUtils.hpp>
#endif

namespace polyfem::basis
{

	enum class ElementKind
	{
		Unknown, // Placeholder for basis relying on opaque callback
		Simplex,
		Quad,
		Hex,
		Prism,
		Pyramid,
	};

	enum class BasisFamily
	{
		Unknown,  // Placeholder for basis relying on opaque callback
		Lagrange, // Lagrange + Berstein.
		Rational
	};

	/// Callback to eval basis value and gradient.
	///
	/// Let N = element basis num.
	///     D = element dimension. Can be 1,2, or 3.
	///     Q = quadrature num.
	/// quad_x/y/z -> size Q quadrature points span. y is empty for D<2. z is empty for D<2.
	/// values     -> size N*Q basis value output.
	///               Layout: [ basis0(q0) basis1(q1) ... ] [ basis1(q0) basis1(q1) ...] ...
	/// grad_x/y/z -> size N*Q basis gradient output. y is empty for D<2. z is empty for D<2.
	///               Layout: [ basis0(q0) basis1(q1) ... ] [ basis1(q0) basis1(q1) ...] ...
	// clang-format off
	using BasisEvalCallback = std::function<void(Span<const double> quad_x,
			                                         Span<const double> quad_y,
			                                         Span<const double> quad_z,
			                                         Span<double> values,
			                                         Span<double> grad_x,
			                                         Span<double> grad_y,
			                                         Span<double> grad_z)>;
	// clang-format on

	struct BasisDesc
	{
		// Common
		ElementKind element_kind;
		BasisFamily basis_family;
		int order;            //< Main basis order.
		int orderq;           //< Secondary basis order for prism.
		int dim;              //< Element dim.
		int basis_num;        //< Element local basis node num.
		int eval_callback_id; //< Optional opaque callback id, -1 implies none.
		int is_parametric;    //< If false, quadrature point lives in rest physical space.

		// Lagrange
		bool is_bernstein;

		// Rational
		Range rational_weight_range;
	};

	struct BasisStoreView
	{
		Span<const double> rational_weights;
		Span<const BasisEvalCallback> eval_callbacks;
	};

	class BasisStore
	{
	private:
		std::vector<double> rational_weights_;
		std::vector<BasisEvalCallback> eval_callbacks_;

#ifdef POLYFEM_WITH_CUDA
		mutable bool need_host_device_sync_ = true;
		mutable DeviceBuf<double> d_rational_weights_;
#endif

	public:
		/// Append rational weights for rational basis and return range.
		Range append_rational_weights(Span<const double> weights);
		/// Append basis eval callback and return id.
		int append_eval_callback(BasisEvalCallback callback);

		BasisStoreView view() const;

#ifdef POLYFEM_WITH_CUDA
		/// Return view on device. Lazily sync data.
		BasisStoreView device_view(ExecutionPolicy policy) const;

		/// Release device storage.
		void clear_device_storage();
#endif
	};

} // namespace polyfem::basis
