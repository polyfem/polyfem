#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#include <map>

#include <units/units.hpp>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/ExecutionPolicy.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/variant>
#endif

namespace polyfem
{
	namespace utils
	{
#ifdef POLYFEM_WITH_CUDA
		class ExpressionValueView
		{
		private:
			// TODO: support matrix and tiny expr.
			using Storage = cuda::std::variant<cuda::std::monostate, double>;
			Storage storage_;

		public:
			/// Construct empty expr view.
			ExpressionValueView() = default;
			/// Construct Scalar expr view.
			explicit ExpressionValueView(double value) : storage_(value) {}

			POLYFEM_BOTH bool is_device_compatible() const
			{
				return !cuda::std::holds_alternative<cuda::std::monostate>(storage_);
			}

			/// Eval expression. Return Nan for empty expression.
			POLYFEM_BOTH double operator()(
				double x,
				double y,
				double z = 0,
				double t = 0,
				int index = -1) const
			{
				if (const double *value = cuda::std::get_if<double>(&storage_))
					return *value;

				// Return Nan for empty view.
				return cuda::std::numeric_limits<double>::quiet_NaN();
			}
		};

		// Must be true to be able to copy to device.
		static_assert(cuda::std::is_trivially_copyable_v<ExpressionValueView>);
#endif

		class ExpressionValue
		{
		public:
			ExpressionValue();

			void set_unit_type(const std::string &unit_type)
			{
				unit_type_ = units::unit_from_string(unit_type);
				unit_type_set_ = true;
				for (auto &expr : mat_expr_)
					expr.set_unit_type(unit_type);
			}

			void init(const json &vals, const std::string &root_path);
			void init(const double val);
			void init(const Eigen::MatrixXd &val);
			void init(const std::string &expr, const std::string &root_path);
#ifdef POLYFEM_WITH_PYTHON
			void init_python(const std::string &path, const std::string &function_name);
#endif

			void init(const std::function<double(double x, double y, double z)> &func);
			void init(const std::function<double(double x, double y, double z, double t)> &func);
			void init(const std::function<double(double x, double y, double z, double t, int index)> &func);

			void init(const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const int coo);
			void init(const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const int coo);

			void set_t(const json &t);

			double operator()(double x, double y, double z = 0, double t = 0, int index = -1) const;

#ifdef POLYFEM_WITH_CUDA
			bool is_device_compatible() const;
			ExpressionValueView device_view(ExecutionPolicy policy) const;
#endif

			void clear();

			bool is_zero() const
			{
				return expr_.empty() && mat_.size() == 0 && mat_expr_.empty() && !sfunc_ && !tfunc_ && fabs(value_) < 1e-10;
			}
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
