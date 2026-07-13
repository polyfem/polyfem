#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <string>
#include <utility>

namespace polyfem::material
{
	enum class LameParamType
	{
		YoungPoisson,
		LambdaMu
	};

	template <typename Scalar>
	struct LameParameter
	{
		LameParamType type = LameParamType::LambdaMu;

		// E/nu or lambda/mu.
		Scalar coeff1;
		Scalar coeff2;
	};

	LameParameter<utils::ExpressionValue> lame_parameter_from_json(
		const json &j,
		const Units &units,
		const std::string &root_path);

	inline LameParameter<double> eval_expr(
		const LameParameter<utils::ExpressionValue> &expr,
		double x,
		double y,
		double z = 0,
		double t = 0,
		int element_id = -1)
	{
		return {expr.type, expr.coeff1(x, y, z, t, element_id), expr.coeff2(x, y, z, t, element_id)};
	}

#ifdef POLYFEM_WITH_CUDA
	inline LameParameter<utils::ExpressionValueView> make_device_expr(
		const LameParameter<utils::ExpressionValue> &expr,
		ExecutionPolicy policy)
	{
		return {
			expr.type,
			expr.coeff1.device_view(policy),
			expr.coeff2.device_view(policy)};
	}

	inline bool is_device_compatible(
		const LameParameter<utils::ExpressionValue> &expr)
	{
		return expr.coeff1.is_device_compatible() && expr.coeff2.is_device_compatible();
	}

	POLYFEM_BOTH inline LameParameter<double> eval_expr(
		const LameParameter<utils::ExpressionValueView> &expr,
		double x,
		double y,
		double z = 0,
		double t = 0,
		int element_id = -1)
	{
		return {expr.type, expr.coeff1(x, y, z, t, element_id), expr.coeff2(x, y, z, t, element_id)};
	}
#endif

	/// Get (lambda, mu) from LameParameter. Convert from E/nu if necessary.
	template <int dim>
	POLYFEM_BOTH std::pair<double, double> lambda_mu(const LameParameter<double> &lame)
	{
		if (lame.type == LameParamType::LambdaMu)
		{
			return {lame.coeff1, lame.coeff2};
		}

		double E = lame.coeff1;
		double nu = lame.coeff2;
		double mu = E / (2.0 * (1.0 + nu));
		if constexpr (dim == 1)
		{
			return {0.0, E / 2.0};
		}
		else if constexpr (dim == 2)
		{
			return {(nu * E) / (1.0 - nu * nu), mu};
		}
		else
		{
			return {(E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)), mu};
		}
	}
} // namespace polyfem::material
