#pragma once

#include <polyfem/Common.hpp>

#include <tinyexpr.h>


namespace polyfem {

	class ExpressionValue
	{
	public:
		~ExpressionValue();
		ExpressionValue();
		void init(const json &vals);
		void init(const double val);
		void init(const std::string &expr);

		double operator()(double x, double y) const;
		double operator()(double x, double y, double z) const;

		bool is_zero() const { return !expr_ && fabs(value_) < 1e-10; }

	private:
		struct Internal
		{
			double x, y, z;
		};

		te_expr *expr_;
		double value_;
		Internal *vals_;
	};

} // namespace polyfem
