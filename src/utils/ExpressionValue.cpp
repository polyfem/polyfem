#include <polyfem/ExpressionValue.hpp>
#include <polyfem/Logger.hpp>

namespace polyfem
{
	ExpressionValue::~ExpressionValue()
	{
		te_free(expr_);
		expr_ = nullptr;
		sfunc_ = nullptr;
		tfunc_ = nullptr;
	}

	ExpressionValue::ExpressionValue()
	{
		expr_ = nullptr;
		value_ = 0;
		vals_ = new Internal();
	}

	void ExpressionValue::clear()
	{
		te_free(expr_);
		expr_ = nullptr;
		sfunc_ = nullptr;
		tfunc_ = nullptr;
		value_ = 0;
	}

	void ExpressionValue::init(const double val)
	{
		te_free(expr_);
		sfunc_ = nullptr;
		tfunc_ = nullptr;
		expr_ = nullptr;

		value_ = val;
	}

	static double max(double a, double b) { return a > b ? a : b; }
	static double min(double a, double b) { return a < b ? a : b; }

	void ExpressionValue::init(const std::string &expr)
	{
		value_ = 0;
		sfunc_ = nullptr;
		tfunc_ = nullptr;
		te_free(expr_);

		if (expr.empty())
		{
			expr_ = nullptr;
			return;
		}

		te_variable vars[] = {
			{"max", (const void *)max, TE_FUNCTION2},
			{"min", (const void *)min, TE_FUNCTION2},
			{"x", &vals_->x},
			{"y", &vals_->y},
			{"z", &vals_->z},
			{"t", &vals_->t},
		};

		int err;
		expr_ = te_compile(expr.c_str(), vars, 6, &err);

		if (!expr_)
		{
			logger().error("Unable to parse {}, error, {}", expr, err);

			assert(false);
		}
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
		te_free(expr_);
		expr_ = nullptr;
		tfunc_ = nullptr;
		value_ = 0;

		sfunc_ = [func](double x, double y, double z, double t) { return func(x, y, z); };
	}

	void ExpressionValue::init(const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const int coo)
	{
		te_free(expr_);
		expr_ = nullptr;
		sfunc_ = nullptr;
		value_ = 0;

		tfunc_ = [func](double x, double y, double z, double t) { return func(x, y, z); };
		tfunc_coo_ = coo;
	}

	void ExpressionValue::init(const std::function<double(double x, double y, double z, double t)> &func)
	{
		te_free(expr_);
		expr_ = nullptr;
		tfunc_ = nullptr;
		value_ = 0;

		sfunc_ = func;
	}

	void ExpressionValue::init(const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const int coo)
	{
		te_free(expr_);
		expr_ = nullptr;
		sfunc_ = nullptr;
		value_ = 0;

		tfunc_ = func;
		tfunc_coo_ = coo;
	}

	double ExpressionValue::operator()(double x, double y, double z, double t) const
	{
		if (!expr_)
		{
			if (sfunc_)
				return sfunc_(x, y, z, t);

			if (tfunc_)
				return tfunc_(x, y, z, t)(tfunc_coo_);

			return value_;
		}

		vals_->x = x;
		vals_->y = y;
		vals_->z = z;
		vals_->t = t;

		return te_eval(expr_);
	}

} // namespace polyfem
