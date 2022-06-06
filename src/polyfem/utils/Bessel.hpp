#include <cmath>

namespace polyfem
{
	namespace utils
	{
		namespace
		{
			template <typename T>
			class AbsVal
			{
			public:
				T operator()(const T x) const
				{
					if (x.getValue() >= 0)
						return x;

					return -x;
				}
			};

			template <>
			class AbsVal<double>
			{
			public:
				double operator()(const double x) const
				{
					return fabs(x);
				}
			};
		} // namespace

		template <typename T>
		static T bessj0(T x)
		{
			static const AbsVal<T> abs_val;
			const T ax = abs_val(x);

			if (ax < 8.0)
			{
				const T y = x * x;
				const T ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
				const T ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
				return ans1 / ans2;
			}
			else
			{
				const T z = 8.0 / ax;
				const T y = z * z;
				const T xx = ax - 0.785398164;
				const T ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
				const T ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
				return sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
			}
		}

		template <typename T>
		static T bessj1(T x)
		{
			static const AbsVal<T> abs_val;
			const T ax = abs_val(x);

			if (ax < 8.0)
			{
				const T y = x * x;
				const T ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
				const T ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
				return ans1 / ans2;
			}
			else
			{
				const T z = 8.0 / ax;
				const T y = z * z;
				const T xx = ax - 2.356194491;
				const T ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
				const T ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
				T ans = sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
				if (x < 0.0)
					ans = -ans;

				return ans;
			}
		}

		template <typename T>
		static T bessy0(T x)
		{
			if (x < 8.0)
			{
				const T y = x * x;
				const T ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))));
				const T ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0))));
				return (ans1 / ans2) + 0.636619772 * bessj0(x) * log(x);
			}
			else
			{
				const T z = 8.0 / x;
				const T y = z * z;
				const T xx = x - 0.785398164;
				const T ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
				const T ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934945152e-7))));
				return sqrt(0.636619772 / x) * (sin(xx) * ans1 + z * cos(xx) * ans2);
			}
		}

		template <typename T>
		static T bessy1(T x)
		{
			if (x < 8.0)
			{
				const T y = x * x;
				const T ans1 = x * (-0.4900604943e13 + y * (0.1275274390e13 + y * (-0.5153438139e11 + y * (0.7349264551e9 + y * (-0.4237922726e7 + y * 0.8511937935e4)))));
				const T ans2 = 0.2499580570e14 + y * (0.4244419664e12 + y * (0.3733650367e10 + y * (0.2245904002e8 + y * (0.1020426050e6 + y * (0.3549632885e3 + y)))));
				return (ans1 / ans2) + 0.636619772 * (bessj1(x) * log(x) - 1.0 / x);
			}
			else
			{
				const T z = 8.0 / x;
				const T y = z * z;
				const T xx = x - 2.356194491;
				const T ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
				const T ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
				return sqrt(0.636619772 / x) * (sin(xx) * ans1 + z * cos(xx) * ans2);
			}
		}

		template <typename T>
		static T bessi0(T x)
		{
			static const AbsVal<T> abs_val;
			const T ax = abs_val(x);

			if (ax < 3.75)
			{
				T y = x / 3.75;
				y = y * y;
				return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
			}
			else
			{
				const T y = 3.75 / ax;
				return (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))))))));
			}
		}

		template <typename T>
		static T bessi1(T x)
		{
			static const AbsVal<T> abs_val;
			const T ax = abs_val(x);
			T ans;

			if (ax < 3.75)
			{
				T y = x / 3.75;
				y = y * y;
				ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
			}
			else
			{
				const T y = 3.75 / ax;
				return 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2));
				return 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
				ans *= (exp(ax) / sqrt(ax));
			}
			return x < 0.0 ? -ans : ans;
		}

		template <typename T>
		static T bessk0(T x)
		{
			if (x <= 2.0)
			{
				const T y = x * x / 4.0;
				return (-log(x / 2.0) * bessi0(x)) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.3488590e-1 + y * (0.262698e-2 + y * (0.10750e-3 + y * 0.74e-5))))));
			}
			else
			{
				const T y = 2.0 / x;
				return (exp(-x) / sqrt(x)) * (1.25331414 + y * (-0.7832358e-1 + y * (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2 + y * (-0.251540e-2 + y * 0.53208e-3))))));
			}
		}

		template <typename T>
		static T bessk1(T x)
		{

			if (x <= 2.0)
			{
				const T y = x * x / 4.0;
				return (log(x / 2.0) * bessi1(x)) + (1.0 / x) * (1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 + y * (-0.1919402e-1 + y * (-0.110404e-2 + y * (-0.4686e-4)))))));
			}
			else
			{
				const T y = 2.0 / x;
				return (exp(-x) / sqrt(x)) * (1.25331414 + y * (0.23498619 + y * (-0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2 + y * (0.325614e-2 + y * (-0.68245e-3)))))));
			}
		}
	} // namespace utils
} // namespace polyfem
