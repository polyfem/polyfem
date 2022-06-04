////////////////////////////////////////////////////////////////////////////////
#include "TestProblem.hpp"
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

namespace polyfem
{
	namespace problem
	{
		namespace
		{

			// -----------------------------------------------------------------------------

			template <typename T>
			inline T pow2(T x) { return x * x; }

			// -----------------------------------------------------------------------------

			template <typename T>
			T reentrant_corner(T x, T y, double omega)
			{
				const double alpha = M_PI / omega;
				const T r = sqrt(x * x + y * y);
				const T theta = atan2(y, x) + (y < 0 ? 2.0 * M_PI : 0.0);
				return pow(r, alpha) * sin(alpha * theta);
			}

			template <typename T>
			std::array<T, 2> linear_elasticity_mode_1(T x, T y, double nu, double E, double lambda, double Q)
			{
				const double kappa = 3.0 - 4.0 * nu;
				const double G = E / (2.0 * (1.0 + nu));
				const T r = sqrt(x * x + y * y);
				const T theta = atan2(y, x) + (y < 0 ? 2.0 * M_PI : 0.0);
				return {{
					1.0 / (2.0 * G) * pow(r, lambda) * ((kappa - Q * (lambda + 1)) * cos(lambda * theta) - lambda * cos((lambda - 2) * theta)),
					1.0 / (2.0 * G) * pow(r, lambda) * ((kappa + Q * (lambda + 1)) * sin(lambda * theta) + lambda * sin((lambda - 2) * theta)),
				}};
			}

			template <typename T>
			std::array<T, 2> linear_elasticity_mode_2(T x, T y, double nu, double E, double lambda, double Q)
			{
				const double kappa = 3.0 - 4.0 * nu;
				const double G = E / (2.0 * (1.0 + nu));
				const T r = sqrt(x * x + y * y);
				const T theta = atan2(y, x) + (y < 0 ? 2.0 * M_PI : 0.0);
				return {{
					1.0 / (2.0 * G) * pow(r, lambda) * ((kappa - Q * (lambda + 1)) * sin(lambda * theta) - lambda * sin((lambda - 2) * theta)),
					1.0 / (2.0 * G) * pow(r, lambda) * ((kappa + Q * (lambda + 1)) * cos(lambda * theta) + lambda * cos((lambda - 2) * theta)),
				}};
			}

			template <typename T>
			T peak(T x, T y, double x_c, double y_c, double alpha)
			{
				return exp(-alpha * (pow2(x - x_c) + pow2(y - y_c)));
			}

			template <typename T>
			T boundary_line_singularity(T x, T y, double alpha)
			{
				return pow(x, alpha);
			}

			template <typename T>
			T wave_front(T x, T y, double x_c, double y_c, double r_0, double alpha)
			{
				const T r = sqrt(pow2(x - x_c) + pow2(y - y_c));
				return atan2(alpha * (r - r_0), T(1.0));
			}

			template <typename T>
			T interior_line_singularity(T x, T y, double alpha, double beta)
			{
				if (x <= beta * (y + 1))
				{
					return cos(M_PI * y / 2);
				}
				else
				{
					return cos(M_PI * y / 2) + pow(x - beta * (y + 1), alpha);
				}
			}

			template <typename T>
			T multiple_difficulties(T x, T y, double omega, double x_w, double y_w, double r_0, double alpha_w,
									double x_p, double y_p, double alpha_p, double eps)
			{
				const T r = sqrt(x * x + y * y);
				const T theta = atan2(y, x) + (y < 0 ? 2.0 * M_PI : 0.0);
				return pow(r, M_PI / omega) * sin(theta * M_PI / omega)
					   + atan2(alpha_w * (sqrt(pow2(x - x_w) + pow2(y - y_w)) - r_0), T(1.0))
					   + exp(-alpha_p * (pow2(x - x_p) + pow2(y - y_p)))
					   + exp(-(1 + y) / eps);
			}

			// -----------------------------------------------------------------------------

		} // anonymous namespace

		////////////////////////////////////////////////////////////////////////////////

#define PARAM(x) (params_[#x].get<double>())

		////////////////////////////////////////////////////////////////////////////////

		TestProblem::TestProblem(const std::string &name)
			: ProblemWithSolution(name)
		{
			params_ = {
				{"type", "reentrant_corner"},
				{"omega", 7.0 * M_PI / 4.0},
				{"is_scalar", true}};
		}

		template <typename T>
		T TestProblem::eval_impl(const T &pt) const
		{
			T res(is_scalar() ? 1 : 2);
			if (params_["type"] == "reentrant_corner")
			{
				res(0) = reentrant_corner(pt(0), pt(1), PARAM(omega));
			}
			else if (params_["type"] == "linear_elasticity_mode_1")
			{
				auto uv = linear_elasticity_mode_1(pt(0), pt(1), PARAM(nu), PARAM(E), PARAM(lambda), PARAM(Q));
				res(0) = uv[0];
				res(1) = uv[1];
			}
			else if (params_["type"] == "linear_elasticity_mode_2")
			{
				auto uv = linear_elasticity_mode_2(pt(0), pt(1), PARAM(nu), PARAM(E), PARAM(lambda), PARAM(Q));
				res(0) = uv[0];
				res(1) = uv[1];
			}
			else if (params_["type"] == "peak")
			{
				res(0) = peak(pt(0), pt(1), PARAM(x_c), PARAM(y_c), PARAM(alpha));
			}
			else if (params_["type"] == "boundary_line_singularity")
			{
				res(0) = boundary_line_singularity(pt(0), pt(1), PARAM(alpha));
			}
			else if (params_["type"] == "wave_front")
			{
				res(0) = wave_front(pt(0), pt(1), PARAM(x_c), PARAM(y_c), PARAM(r_0), PARAM(alpha));
			}
			else if (params_["type"] == "interior_line_singularity")
			{
				res(0) = interior_line_singularity(pt(0), pt(1), PARAM(alpha), PARAM(beta));
			}
			else if (params_["type"] == "multiple_difficulties")
			{
				res(0) = multiple_difficulties(pt(0), pt(1),
											   PARAM(omega), PARAM(x_w), PARAM(y_w), PARAM(r_0), PARAM(alpha_w),
											   PARAM(x_p), PARAM(y_p), PARAM(alpha_p), PARAM(eps));
			}
			return res;
		}

		void TestProblem::set_parameters(const json &params)
		{
			// j_original.update(j_patch);
			assert(!params.is_null());
			params_.merge_patch(params);
		}
	} // namespace problem
} // namespace polyfem
