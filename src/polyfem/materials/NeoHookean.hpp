#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/materials/LameParameter.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <string>

namespace polyfem::material
{
	template <typename Scalar>
	struct NeoHookean
	{
		using ExprType = NeoHookean<utils::ExpressionValue>;
#ifdef POLYFEM_WITH_CUDA
		using ExprViewType = NeoHookean<utils::ExpressionValueView>;
#endif

		LameParameter<Scalar> lame;
	};

	NeoHookean<utils::ExpressionValue> neo_hookean_from_json(const json &j, const Units &units, const std::string &root_path);

	//------------------------------------------------------------
	// All material must overload below functions
	//------------------------------------------------------------

	inline NeoHookean<double> eval_expr(
		const NeoHookean<utils::ExpressionValue> &expr,
		double x,
		double y,
		double z = 0,
		double t = 0,
		int element_id = -1)
	{
		return {eval_expr(expr.lame, x, y, z, t, element_id)};
	}

#ifdef POLYFEM_WITH_CUDA
	inline NeoHookean<utils::ExpressionValueView> make_device_expr(const NeoHookean<utils::ExpressionValue> &expr, ExecutionPolicy policy)
	{
		return NeoHookean<utils::ExpressionValueView>{make_device_expr(expr.lame, policy)};
	}

	inline bool is_device_compatible(const NeoHookean<utils::ExpressionValue> &expr)
	{
		return is_device_compatible(expr.lame);
	}

	POLYFEM_BOTH inline NeoHookean<double> eval_expr(
		const NeoHookean<utils::ExpressionValueView> &expr,
		double x,
		double y,
		double z = 0,
		double t = 0,
		int element_id = -1)
	{
		return {eval_expr(expr.lame, x, y, z, t, element_id)};
	}
#endif
} // namespace polyfem::material
