#include <polyfem/ExpressionValue.hpp>
#include <polyfem/Logger.hpp>

namespace polyfem
{
	ExpressionValue::~ExpressionValue()
	{
		te_free(expr_);
		delete vals_;
	}

	ExpressionValue::ExpressionValue()
	{
		expr_ = nullptr;
		value_ = 0;
		vals_ = new Internal();
	}

	void ExpressionValue::init(const double val)
	{
		te_free(expr_);
		expr_ = nullptr;

		value_ = val;
	}


	void ExpressionValue::init(const std::string &expr)
	{
		value_ = 0;
		te_free(expr_);

		if(expr.empty())
		{
			expr_ = nullptr;
			return;
		}

		te_variable vars[3];
		vars[0] = {"x", &vals_->x};
		vars[1] = {"y", &vals_->y};
		vars[2] = {"z", &vals_->z};

		int err;
		expr_ = te_compile(expr.c_str(), vars, 3, &err);

		if(!expr_)
		{
			logger().error("Unable to parse {}, error, {}", expr, err);

			assert(false);
		}
	}

	void ExpressionValue::init(const json &vals)
	{
		if(vals.is_number())
		{
			init((double)vals);
		}
		else
		{
			const std::string expr = vals;
			init(expr);
		}
	}

	double ExpressionValue::operator()(double x, double y) const
	{
		if(!expr_)
			return value_;

		vals_->x = x;
		vals_->y = y;

		return te_eval(expr_);
	}

	double ExpressionValue::operator()(double x, double y, double z) const
	{
		if(!expr_)
			return value_;

		vals_->x = x;
		vals_->y = y;
		vals_->z = z;

		return te_eval(expr_);
	}

}