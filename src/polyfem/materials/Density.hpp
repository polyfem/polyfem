#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <string>

namespace polyfem::material
{
	template <typename Scalar>
	struct Density
	{
		using ExprType = Density<utils::ExpressionValue>;
#ifdef POLYFEM_WITH_CUDA
		using ExprViewType = Density<utils::ExpressionValueView>;
#endif

		Scalar rho;
	};

	Density<utils::ExpressionValue> density_from_json(const json &, const Units &, const std::string &);

	//------------------------------------------------------------
	// All material must overload below functions
	//------------------------------------------------------------

	/// @brief Evaluate material expression.
	/// @param expr Material expression.
	/// @param x Quadrature point x.
	/// @param y Quadrature point y.
	/// @param z Quadrature point z.
	/// @param t Time.
	/// @param element_id
	inline Density<double> eval_expr(
		const Density<utils::ExpressionValue> &expr,
		double x,
		double y,
		double z = 0,
		double t = 0,
		int element_id = -1)
	{
		Density<double> out{};
		out.rho = expr.rho(x, y, z, t, element_id);
		return out;
	}

#ifdef POLYFEM_WITH_CUDA
	inline Density<utils::ExpressionValueView> make_device_expr(const Density<utils::ExpressionValue> &expr, ExecutionPolicy policy)
	{
		return Density<utils::ExpressionValueView>{expr.rho.device_view(policy)};
	}

	inline bool is_device_compatible(const Density<utils::ExpressionValue> &expr)
	{
		return expr.rho.is_device_compatible();
	}

	/// @brief Evaluate material expression view.
	/// @param expr Material expression.
	/// @param x Quadrature point x.
	/// @param y Quadrature point y.
	/// @param z Quadrature point z.
	/// @param t Time.
	/// @param element_id
	POLYFEM_BOTH inline Density<double> eval_expr(
		const Density<utils::ExpressionValueView> &expr,
		double x,
		double y,
		double z = 0,
		double t = 0,
		int element_id = -1)
	{
		return Density<double>{expr.rho(x, y, z, t, element_id)};
	}
#endif
} // namespace polyfem::material
