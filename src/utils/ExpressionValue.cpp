#include <polyfem/ExpressionValue.hpp>
#include <polyfem/Logger.hpp>

namespace polyfem
{
	ExpressionValue::~ExpressionValue()
	{
		te_free(expr_);
		expr_ = nullptr;
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
		sfunc_ = nullptr;
		tfunc_ = nullptr;
		expr_ = nullptr;

		value_ = val;
	}

	void ExpressionValue::init(const std::string &expr)
	{
		value_ = 0;
		sfunc_ = nullptr;
		tfunc_ = nullptr;
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

	void ExpressionValue::init(const std::function<double(double x, double y, double z)> &func)
	{
		te_free(expr_);
		expr_ = nullptr;
		tfunc_ = nullptr;
		value_ = 0;

		sfunc_ = func;
	}

	void ExpressionValue::init(const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const int coo)
	{
		te_free(expr_);
		expr_ = nullptr;
		sfunc_ = nullptr;
		value_ = 0;

		tfunc_ = func;
		tfunc_coo_ = coo;
	}

	double ExpressionValue::operator()(double x, double y) const
	{
		if(!expr_){
			if(sfunc_)
				return sfunc_(x, y, 0);

			if (tfunc_)
				return tfunc_(x, y, 0)(tfunc_coo_);

			return value_;
		}

		vals_->x = x;
		vals_->y = y;

		return te_eval(expr_);
	}

	double ExpressionValue::operator()(double x, double y, double z) const
	{
		if (!expr_)
		{
			if (sfunc_)
				return sfunc_(x, y, z);

			if (tfunc_)
				return tfunc_(x, y, z)(tfunc_coo_);

			return value_;
		}

		vals_->x = x;
		vals_->y = y;
		vals_->z = z;

		return te_eval(expr_);
	}

}