#pragma once

#include <polyfem/Common.hpp>

#include <units/units.hpp>

namespace polyfem
{
	namespace utils
	{
		class ExpressionValue
		{
		public:
			ExpressionValue();

			void set_unit_type(const std::string &unit_type)
			{
				unit_type_ = units::unit_from_string(unit_type);
				unit_type_set_ = true;
			}

			void init(const json &vals);
			void init(const double val);
			void init(const Eigen::MatrixXd &val);
			void init(const std::string &expr);

			void init(const std::function<double(double x, double y, double z)> &func);
			void init(const std::function<double(double x, double y, double z, double t)> &func);
			void init(const std::function<double(double x, double y, double z, double t, int index)> &func);

			void init(const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const int coo);
			void init(const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const int coo);

			double operator()(double x, double y, double z = 0, double t = 0, int index = -1) const;

			void clear();

			bool is_zero() const { return expr_.empty() && fabs(value_) < 1e-10; }

		private:
			std::function<double(double x, double y, double z, double t, int index)> sfunc_;
			std::function<Eigen::MatrixXd(double x, double y, double z, double t)> tfunc_;
			int tfunc_coo_;

			std::string expr_;
			double value_;
			Eigen::MatrixXd mat_;

			units::precise_unit unit_type_;
			units::precise_unit unit_;
			bool unit_type_set_ = false;
		};
	} // namespace utils
} // namespace polyfem
