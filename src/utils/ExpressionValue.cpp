#include <polyfem/ExpressionValue.hpp>
#include <polyfem/Logger.hpp>

#include <igl/PI.h>

namespace polyfem
{
	ExpressionValue::~ExpressionValue()
	{
		sfunc_ = nullptr;
		tfunc_ = nullptr;
	}

	ExpressionValue::ExpressionValue()
	{
		expr_ = "";
		value_ = 0;
	}

	void ExpressionValue::clear()
	{
		expr_ = "";
		sfunc_ = nullptr;
		tfunc_ = nullptr;
		value_ = 0;
	}

	void ExpressionValue::init(const double val)
	{
		sfunc_ = nullptr;
		tfunc_ = nullptr;
		expr_ = "";

		value_ = val;
	}

	static double min(double a, double b) { return a < b ? a : b; }
	static double max(double a, double b) { return a > b ? a : b; }
	static double deg2rad(double d) { return d * igl::PI / 180.0; }
	static double rotate_2D_x(double x, double y, double theta)
	{
		return x * cos(theta) - y * sin(theta);
	}
	static double rotate_2D_y(double x, double y, double theta)
	{
		return x * sin(theta) + y * cos(theta);
	}
	static double smooth_abs(double x, double k)
	{
		return tanh(k * x) * x;
	}

	void ExpressionValue::init(const std::string &expr)
	{
		value_ = 0;
		sfunc_ = nullptr;
		tfunc_ = nullptr;

		expr_ = expr;

		if (expr.empty())
		{
			return;
		}

		double x = 0, y = 0, z = 0, t = 0;

		std::vector<te_variable> vars = {
			{"x", &x, TE_VARIABLE},
			{"y", &y, TE_VARIABLE},
			{"z", &z, TE_VARIABLE},
			{"t", &t, TE_VARIABLE},
			{"min", (const void *)min, TE_FUNCTION2},
			{"max", (const void *)max, TE_FUNCTION2},
			{"deg2rad", (const void *)deg2rad, TE_FUNCTION1},
			{"rotate_2D_x", (const void *)rotate_2D_x, TE_FUNCTION3},
			{"rotate_2D_y", (const void *)rotate_2D_y, TE_FUNCTION3},
			{"smooth_abs", (const void *)smooth_abs, TE_FUNCTION2},
		};

		int err;
		te_expr *tmp = te_compile(expr.c_str(), vars.data(), vars.size(), &err);
		if (!tmp)
		{
			logger().error("Unable to parse: {}", expr);
			logger().error("Error near here: {0: >{1}}", "^", err - 1);
			assert(false);
		}
		te_free(tmp);
	}

	void ExpressionValue::init(const json &vals)
	{
		if (vals.is_number())
		{
			init(vals.get<double>());
		}
		else
		{
			init(vals.get<std::string>());
		}
	}

	void ExpressionValue::init(const std::function<double(double x, double y, double z)> &func)
	{
		expr_ = "";
		tfunc_ = nullptr;
		value_ = 0;

		sfunc_ = [func](double x, double y, double z, double t) { return func(x, y, z); };
	}

	void ExpressionValue::init(const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const int coo)
	{
		expr_ = "";
		sfunc_ = nullptr;
		value_ = 0;

		tfunc_ = [func](double x, double y, double z, double t) { return func(x, y, z); };
		tfunc_coo_ = coo;
	}

	void ExpressionValue::init(const std::function<double(double x, double y, double z, double t)> &func)
	{
		expr_ = "";
		tfunc_ = nullptr;
		value_ = 0;

		sfunc_ = func;
	}

	void ExpressionValue::init(const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const int coo)
	{
		expr_ = "";
		sfunc_ = nullptr;
		value_ = 0;

		tfunc_ = func;
		tfunc_coo_ = coo;
	}

	double ExpressionValue::operator()(double x, double y, double z, double t) const
	{
		if (expr_.empty())
		{
			if (sfunc_)
				return sfunc_(x, y, z, t);

			if (tfunc_)
				return tfunc_(x, y, z, t)(tfunc_coo_);

			return value_;
		}

		std::vector<te_variable> vars = {
			{"x", &x, TE_VARIABLE},
			{"y", &y, TE_VARIABLE},
			{"z", &z, TE_VARIABLE},
			{"t", &t, TE_VARIABLE},
			{"min", (const void *)min, TE_FUNCTION2},
			{"max", (const void *)max, TE_FUNCTION2},
			{"deg2rad", (const void *)deg2rad, TE_FUNCTION1},
			{"rotate_2D_x", (const void *)rotate_2D_x, TE_FUNCTION3},
			{"rotate_2D_y", (const void *)rotate_2D_y, TE_FUNCTION3},
			{"smooth_abs", (const void *)smooth_abs, TE_FUNCTION2},
		};

		int err;
		te_expr *tmp = te_compile(expr_.c_str(), vars.data(), vars.size(), &err);
		assert(tmp != nullptr);
		const auto res = te_eval(tmp);
		te_free(tmp);

		return res;
	}

} // namespace polyfem
