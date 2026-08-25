#include "ExpressionValue.hpp"

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <units/units.hpp>

#include <igl/PI.h>

#include <tinyexpr.h>
#include <filesystem>
#include <memory>
#ifdef POLYFEM_WITH_PYTHON
// pybind11 enables a debug-only reference counter when NDEBUG is not defined.
// On MSVC that path can fail with C2480 because it uses a function-local
// thread_local variable. Keep the workaround scoped to pybind11's headers.
#if defined(_MSC_VER) && !defined(NDEBUG) && !defined(PYBIND11_HANDLE_REF_DEBUG)
#define POLYFEM_RESTORE_NDEBUG_AFTER_PYBIND11
#define NDEBUG
#endif
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef POLYFEM_RESTORE_NDEBUG_AFTER_PYBIND11
#undef NDEBUG
#undef POLYFEM_RESTORE_NDEBUG_AFTER_PYBIND11
#endif
#endif

#include <iostream>

namespace polyfem
{
	using namespace io;

	namespace utils
	{
#ifdef POLYFEM_WITH_PYTHON
		namespace py = pybind11;

		namespace
		{
			class PythonInterpreter
			{
			public:
				static PythonInterpreter &instance()
				{
					static PythonInterpreter interpreter;
					return interpreter;
				}

				PythonInterpreter(const PythonInterpreter &) = delete;
				PythonInterpreter &operator=(const PythonInterpreter &) = delete;

			private:
				PythonInterpreter()
				{
					if (!Py_IsInitialized())
					{
						// Keep the embedded interpreter alive for the process lifetime.
						// Python-backed ExpressionValues may be destroyed during static
						// teardown, and finalizing first makes their py::object cleanup unsafe.
#ifdef POLYFEM_PYTHON_EXECUTABLE
						PyConfig config;
						PyConfig_InitPythonConfig(&config);

						wchar_t *program_raw = Py_DecodeLocale(POLYFEM_PYTHON_EXECUTABLE, nullptr);
						if (program_raw == nullptr)
						{
							PyConfig_Clear(&config);
							log_and_throw_error("Failed to decode Python executable path.");
						}

						PyStatus status = PyConfig_SetString(&config, &config.program_name, program_raw);
						PyMem_RawFree(program_raw);
						if (PyStatus_Exception(status))
						{
							PyConfig_Clear(&config);
							log_and_throw_error("Failed to configure embedded Python program_name.");
						}

						guard_ = new py::scoped_interpreter(&config);
#else
						guard_ = new py::scoped_interpreter();
#endif
					}
				}

				~PythonInterpreter() = default;

				py::scoped_interpreter *guard_ = nullptr;
			};

			void ensure_python_interpreter()
			{
				try
				{
					(void)PythonInterpreter::instance();
				}
				catch (const std::exception &e)
				{
					log_and_throw_error(fmt::format("Failed to initialize embedded Python: {}", e.what()));
				}
			}

			std::shared_ptr<py::object> make_python_object_holder(py::object value)
			{
				return std::shared_ptr<py::object>(
					new py::object(std::move(value)),
					[](py::object *object) {
						if (Py_IsInitialized())
						{
							py::gil_scoped_acquire gil;
							delete object;
						}
						else
						{
							(void)object->release();
							delete object;
						}
					});
			}

			py::object load_python_value_function(const std::string &path, const std::string &function_name)
			{
				ensure_python_interpreter();
				py::gil_scoped_acquire gil;

				py::module_ importlib_util = py::module_::import("importlib.util");
				py::module_ pathlib = py::module_::import("pathlib");

				py::object resolved_path = pathlib.attr("Path")(path).attr("resolve")();
				std::string module_name = resolved_path.attr("stem").cast<std::string>();
				std::string resolved_path_str = py::str(resolved_path).cast<std::string>();

				py::object spec = importlib_util.attr("spec_from_file_location")(module_name, resolved_path_str);
				if (spec.is_none())
					log_and_throw_error(fmt::format("Unable to create Python module spec from '{}'", resolved_path_str));

				py::object loader = spec.attr("loader");
				if (loader.is_none())
					log_and_throw_error(fmt::format("Python module '{}' has no loader", resolved_path_str));

				py::object module = importlib_util.attr("module_from_spec")(spec);
				loader.attr("exec_module")(module);

				if (!py::hasattr(module, function_name.c_str()))
					log_and_throw_error(fmt::format("Python expression file '{}' must define a function named '{}'", resolved_path_str, function_name));

				py::object value = module.attr(function_name.c_str());
				if (!PyCallable_Check(value.ptr()))
					log_and_throw_error(fmt::format("Python attribute '{}' in '{}' is not callable", function_name, resolved_path_str));

				return value;
			}
		} // namespace

		void ExpressionValue::init_python(const std::string &path, const std::string &function_name)
		{
			clear();

			py::object value = load_python_value_function(path, function_name);
			auto callable = make_python_object_holder(std::move(value));

			sfunc_ = [callable](double x, double y, double z, double t, int index) -> double {
				py::gil_scoped_acquire gil;
				py::object out = (*callable)(x, y, z, t, index);

				try
				{
					return out.cast<double>();
				}
				catch (const py::cast_error &)
				{
					log_and_throw_error("Python expression must return a scalar convertible to double");
					return 0;
				}
			};
		}

#endif

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

		void ExpressionValue::init(const std::string &expr, const std::string &root_path)
		{
			clear();

			if (expr.empty())
			{
				return;
			}

			const auto path = std::filesystem::path(utils::resolve_path(expr, root_path));

			try
			{
				if (std::filesystem::is_regular_file(path))
				{
					read_matrix(path.string(), mat_);
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
				logger().error("Error near character {}.", err);
				log_and_throw_error("Invalid expression '{}'.", expr);
			}
			te_free(tmp);
		}

		void ExpressionValue::init(const json &vals, const std::string &root_path)
		{
			clear();

			if (vals.is_number())
			{
				init(vals.get<double>());
			}
			else if (vals.is_array())
			{
				if (vals.empty() || vals[0].is_number())
				{
					mat_.resize(vals.size(), 1);

					for (int i = 0; i < mat_.size(); ++i)
					{
						if (!vals[i].is_number())
							log_and_throw_error("Expression arrays must contain either only numbers or only expressions.");
						mat_(i) = vals[i].get<double>();
					}
				}
				else
				{
					mat_expr_ = std::vector<ExpressionValue>(vals.size());

					for (int i = 0; i < vals.size(); ++i)
					{
						mat_expr_[i].init(vals[i], root_path);
						if (unit_type_set_)
						{
							mat_expr_[i].unit_type_ = unit_type_;
							mat_expr_[i].unit_type_set_ = true;
						}
					}
				}

				if (t_index_.size() > 0)
					if (mat_.size() != t_index_.size() && mat_expr_.size() != t_index_.size())
						logger().error("Specifying varying dirichlet over time, however 'time_reference' does not match dirichlet boundary conditions.");
			}
			else if (vals.is_object())
			{
				units::precise_unit unit;
				if (vals.contains("unit"))
				{
					if (!vals["unit"].is_string())
						log_and_throw_error("Expression object 'unit' must be a string.");

					const std::string unit_str = vals["unit"].get<std::string>();
					if (!unit_str.empty())
						unit = units::unit_from_string(unit_str);
				}

				if (vals.contains("file_name"))
				{
#ifndef POLYFEM_WITH_PYTHON
					log_and_throw_error(
						"Python expression '{}' requested, but PolyFEM was built without Python support. "
						"Reconfigure with -DPOLYFEM_WITH_PYTHON=ON to enable Python expressions.",
						vals["file_name"].dump());
#else
					if (!vals["file_name"].is_string())
						log_and_throw_error("Python expression 'file_name' must be a string.");
					if (!vals.contains("function_name") || !vals["function_name"].is_string())
						log_and_throw_error("Python expression '{}' must include a string 'function_name'.", vals["file_name"].get<std::string>());

					const std::string path = utils::resolve_path(vals["file_name"].get<std::string>(), root_path);
					const std::string function_name = vals["function_name"].get<std::string>();

					init_python(path, function_name);
					unit_ = unit;
					return;
#endif
				}

				if (!vals.contains("value"))
					log_and_throw_error("Expression object must contain either 'value' or 'file_name'.");

				init(vals["value"], root_path);
				unit_ = unit;
			}
			else
			{
				init(vals.get<std::string>(), root_path);
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
			sfunc_ = [func](double x, double y, double z, double t, int index) { return func(x, y, z, t); };
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
				if (!tmp)
				{
					logger().error("Unable to parse: {}", expr_);
					logger().error("Error near character {}.", err);
					log_and_throw_error("Invalid expression '{}'.", expr_);
				}
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
