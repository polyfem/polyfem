#include "ExpressionValue.hpp"

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/Logger.hpp>

#include <units/units.hpp>

#include <igl/PI.h>

#include <tinyexpr.h>
#include <filesystem>

#include <iostream>

namespace polyfem
{
	using namespace io;

	namespace utils
	{
		static double min(double a, double b) { return a < b ? a : b; }
		static double max(double a, double b) { return a > b ? a : b; }
		static double smoothstep(double a)
		{
			if (a < 0)
				return 0;
			else if (a > 1)
				return 1;
			else
				return (3 * pow(a, 2)) - (2 * pow(a, 3));
		}
		static double half_smoothstep(double a)
		{
			double b = (a + 1.) / 2.;
			return 2. * smoothstep(b) - 1.;
		}
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
		static double iflargerthanzerothenelse(double check, double ttrue, double ffalse)
		{
			return check >= 0 ? ttrue : ffalse;
		}
		static double sign(double x)
		{
			return (0 < x) - (x < 0);
		}
		static double compare(double a, double b)
		{
			return a < b ? 1.0 : 0.0;
		}

		ExpressionValue::ExpressionValue()
		{
			clear();
		}

		void ExpressionValue::clear()
		{
			expr_ = "";
			mat_.resize(0, 0);
			mat_expr_ = {};
			sfunc_ = nullptr;
			tfunc_ = nullptr;
			value_ = 0;
		}

		void ExpressionValue::init(const double val)
		{
			clear();

			value_ = val;
		}

		void ExpressionValue::init(const Eigen::MatrixXd &val)
		{
			clear();

			mat_ = val;
		}

		void ExpressionValue::init(const std::string &expr)
		{
			clear();

			if (expr.empty())
			{
				return;
			}

			const auto path = std::filesystem::path(expr);

			try
			{
				/* code */
				if (std::filesystem::is_regular_file(path))
				{
					read_matrix(expr, mat_);
					return;
				}
			}
			catch (const std::filesystem::filesystem_error &e)
			{
			}

			expr_ = expr;

			double x = 0, y = 0, z = 0, t = 0;

			std::vector<te_variable> vars = {
				{"x", &x, TE_VARIABLE},
				{"y", &y, TE_VARIABLE},
				{"z", &z, TE_VARIABLE},
				{"t", &t, TE_VARIABLE},
				{"min", (const void *)min, TE_FUNCTION2},
				{"max", (const void *)max, TE_FUNCTION2},
				{"smoothstep", (const void *)smoothstep, TE_FUNCTION1},
				{"half_smoothstep", (const void *)half_smoothstep, TE_FUNCTION1},
				{"deg2rad", (const void *)deg2rad, TE_FUNCTION1},
				{"rotate_2D_x", (const void *)rotate_2D_x, TE_FUNCTION3},
				{"rotate_2D_y", (const void *)rotate_2D_y, TE_FUNCTION3},
				{"if", (const void *)iflargerthanzerothenelse, TE_FUNCTION3},
				{"compare", (const void *)compare, TE_FUNCTION2},
				{"smooth_abs", (const void *)smooth_abs, TE_FUNCTION2},
				{"sign", (const void *)sign, TE_FUNCTION1},
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
			clear();

			if (vals.is_number())
			{
				init(vals.get<double>());
			}
			else if (vals.is_array())
			{
				mat_.resize(vals.size(), 1);

				for (int i = 0; i < mat_.size(); ++i)
				{
					if (vals[i].is_string())
						break;
					mat_(i) = vals[i];
				}

				if (vals.size() > 0 && vals[0].is_string())
				{
					mat_.resize(0, 0);
					mat_expr_ = std::vector<ExpressionValue>(vals.size());

					for (int i = 0; i < vals.size(); ++i)
					{
						mat_expr_[i].init(vals[i]);
					}
				}

				if (t_index_.size() > 0)
					if (mat_.size() != t_index_.size() && mat_expr_.size() != t_index_.size())
						logger().error("Specifying varying dirichlet over time, however 'time_reference' does not match dirichlet boundary conditions.");
			}
			else if (vals.is_object())
			{

				unit_ = units::unit_from_string(vals["unit"].get<std::string>());
				init(vals["value"]);
			}
			else
			{
				init(vals.get<std::string>());
			}
		}

		void ExpressionValue::init(const std::function<double(double x, double y, double z)> &func)
		{
			clear();

			sfunc_ = [func](double x, double y, double z, double t, int index) { return func(x, y, z); };
		}

		void ExpressionValue::init(const std::function<double(double x, double y, double z, double t)> &func)
		{
			clear();
			sfunc_ = [func](double x, double y, double z, double t, double index) { return func(x, y, z, t); };
		}

		void ExpressionValue::init(const std::function<double(double x, double y, double z, double t, int index)> &func)
		{
			clear();
			sfunc_ = func;
		}

		void ExpressionValue::init(const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const int coo)
		{
			clear();

			tfunc_ = [func](double x, double y, double z, double t) { return func(x, y, z); };
			tfunc_coo_ = coo;
		}

		void ExpressionValue::init(const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const int coo)
		{
			clear();

			tfunc_ = func;
			tfunc_coo_ = coo;
		}

		void ExpressionValue::set_t(const json &t)
		{
			if (t.is_array())
			{
				for (int i = 0; i < t.size(); ++i)
				{
					t_index_[std::round(t[i].get<double>() * 1000.) / 1000.] = i;
				}

				if (mat_.size() != t_index_.size() && mat_expr_.size() != t_index_.size())
					logger().error("Specifying varying dirichlet over time, however 'time_reference' does not match dirichlet boundary conditions.");
			}
		}

		double ExpressionValue::operator()(double x, double y, double z, double t, int index) const
		{
			assert(unit_type_set_);

			double result;
			if (expr_.empty())
			{
				if (t_index_.size() > 0)
				{
					t = std::round(t * 1000.) / 1000.;
					if (t_index_.count(t) != 0)
					{
						if (mat_.size() > 0)
							return mat_(t_index_.at(t));
						else if (mat_expr_.size() > 0)
							return mat_expr_[t_index_.at(t)](x, y, z, t, index);
					}
					else
					{
						logger().error("Cannot find dirichlet condition for time step {}.", t);
						return 0;
					}
				}

				if (mat_.size() > 0)
					result = mat_(index);
				else if (sfunc_)
					result = sfunc_(x, y, z, t, index);
				else if (tfunc_)
					result = tfunc_(x, y, z, t)(tfunc_coo_);
				else
					result = value_;
			}
			else
			{

				std::vector<te_variable> vars = {
					{"x", &x, TE_VARIABLE},
					{"y", &y, TE_VARIABLE},
					{"z", &z, TE_VARIABLE},
					{"t", &t, TE_VARIABLE},
					{"min", (const void *)min, TE_FUNCTION2},
					{"max", (const void *)max, TE_FUNCTION2},
					{"smoothstep", (const void *)smoothstep, TE_FUNCTION1},
					{"half_smoothstep", (const void *)half_smoothstep, TE_FUNCTION1},
					{"deg2rad", (const void *)deg2rad, TE_FUNCTION1},
					{"rotate_2D_x", (const void *)rotate_2D_x, TE_FUNCTION3},
					{"rotate_2D_y", (const void *)rotate_2D_y, TE_FUNCTION3},
					{"if", (const void *)iflargerthanzerothenelse, TE_FUNCTION3},
					{"compare", (const void *)compare, TE_FUNCTION2},
					{"smooth_abs", (const void *)smooth_abs, TE_FUNCTION2},
					{"sign", (const void *)sign, TE_FUNCTION1},
				};

				int err;
				te_expr *tmp = te_compile(expr_.c_str(), vars.data(), vars.size(), &err);
				assert(tmp != nullptr);
				result = te_eval(tmp);
				te_free(tmp);
			}

			if (!unit_.base_units().empty())
			{
				if (!unit_.is_convertible(unit_type_))
					log_and_throw_error(fmt::format("Cannot convert {} to {}", units::to_string(unit_), units::to_string(unit_type_)));

				result = units::convert(result, unit_, unit_type_);
			}

			return result;
		}
	} // namespace utils
} // namespace polyfem
