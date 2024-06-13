#pragma once

#include <polyfem/Common.hpp>
#include <map>

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

			void set_t(const json &t);

			double operator()(double x, double y, double z = 0, double t = 0, int index = -1) const;

			void clear();

			bool is_zero() const { return expr_.empty() && fabs(value_) < 1e-10; }
			bool is_mat() const
			{
				if (expr_.empty() && mat_.size() > 0)
					return true;
				return false;
			}

			const Eigen::MatrixXd &get_mat() const
			{
				assert(is_mat());
				return mat_;
			}

			void set_mat(const Eigen::MatrixXd &mat)
			{
				assert(is_mat());
				assert(mat_.rows() == mat.rows());
				assert(mat_.cols() == mat.cols());
				mat_ = mat;
			}

			double get_val() const
			{
				return value_;
			}

		private:
			std::function<double(double x, double y, double z, double t, int index)> sfunc_;
			std::function<Eigen::MatrixXd(double x, double y, double z, double t)> tfunc_;
			int tfunc_coo_;

			std::string expr_;
			double value_;
			Eigen::MatrixXd mat_;
			std::vector<ExpressionValue> mat_expr_;
			std::map<double, int> t_index_;

			units::precise_unit unit_type_;
			units::precise_unit unit_;
			bool unit_type_set_ = false;
		};
	} // namespace utils
} // namespace polyfem
