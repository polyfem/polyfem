#pragma once

#include <polyfem/Common.hpp>

#include <tinyexpr.h>

namespace polyfem
{

	class ExpressionValue
	{
	public:
		~ExpressionValue();
		ExpressionValue();
		void init(const json &vals);
		void init(const double val);
		void init(const std::string &expr);

		void init(const std::function<double(double x, double y, double z)> &func);
		void init(const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const int coo);
		void init(const std::function<double(double x, double y, double z, double t)> &func);
		void init(const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const int coo);

		double operator()(double x, double y, double z = 0, double t = 0) const;

		void clear();

		bool is_zero() const { return !expr_ && fabs(value_) < 1e-10; }

	private:
		struct Internal
		{
			double x, y, z, t;
		};

		std::function<double(double x, double y, double z, double t)> sfunc_;
		std::function<Eigen::MatrixXd(double x, double y, double z, double t)> tfunc_;
		int tfunc_coo_;

		te_expr *expr_;
		double value_;
		Internal *vals_;
	};

} // namespace polyfem
