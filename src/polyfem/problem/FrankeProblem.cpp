#include "FrankeProblem.hpp"

#include <iostream>

namespace polyfem
{
	namespace problem
	{
		namespace
		{
			template <typename T>
			T franke_fun(T x, T y)
			{
				auto cx2 = (9 * x - 2) * (9 * x - 2);
				auto cy2 = (9 * y - 2) * (9 * y - 2);

				auto cx1 = (9 * x + 1) * (9 * x + 1);
				auto cx7 = (9 * x - 7) * (9 * x - 7);

				auto cy3 = (9 * y - 3) * (9 * y - 3);
				auto cx4 = (9 * x - 4) * (9 * x - 4);

				auto cy7 = (9 * y - 7) * (9 * y - 7);

				return (3. / 4.) * exp(-(1. / 4.) * cx2 - (1. / 4.) * cy2) + (3. / 4.) * exp(-(1. / 49.) * cx1 - (9. / 10.) * y - 1. / 10.) + (1. / 2.) * exp(-(1. / 4.) * cx7 - (1. / 4.) * cy3) - (1. / 5.) * exp(-cx4 - cy7);
			}

			template <typename T>
			T franke_fun(T x, T y, T z)
			{
				auto cx2 = (9 * x - 2) * (9 * x - 2);
				auto cy2 = (9 * y - 2) * (9 * y - 2);
				auto cz2 = (9 * z - 2) * (9 * z - 2);

				auto cx1 = (9 * x + 1) * (9 * x + 1);
				auto cx7 = (9 * x - 7) * (9 * x - 7);

				auto cy3 = (9 * y - 3) * (9 * y - 3);
				auto cx4 = (9 * x - 4) * (9 * x - 4);
				auto cy7 = (9 * y - 7) * (9 * y - 7);

				auto cz5 = (9 * z - 5) * (9 * z - 5);

				return 3. / 4. * exp(-1. / 4. * cx2 - 1. / 4. * cy2 - 1. / 4. * cz2) + 3. / 4. * exp(-1. / 49. * cx1 - 9. / 10. * y - 1. / 10. - 9. / 10. * z - 1. / 10.) + 1. / 2. * exp(-1. / 4. * cx7 - 1. / 4. * cy3 - 1. / 4. * cz5) - 1. / 5. * exp(-cx4 - cy7 - cz5);
			}

			template <typename T>
			T franke_fun_old(T x, T y, T z)
			{
				auto cx2 = (9 * x - 2) * (9 * x - 2);
				auto cy2 = (9 * y - 2) * (9 * y - 2);
				auto cz2 = (9 * z - 2) * (9 * z - 2);

				auto cx1 = (9 * x + 1) * (9 * x + 1);
				auto cx7 = (9 * x - 7) * (9 * x - 7);

				auto cy3 = (9 * y - 3) * (9 * y - 3);
				auto cx4 = (9 * x - 4) * (9 * x - 4);
				auto cy7 = (9 * y - 7) * (9 * y - 7);

				auto cz5 = (9 * y - 5) * (9 * y - 5);

				return 3. / 4. * exp(-1. / 4. * cx2 - 1. / 4. * cy2 - 1. / 4. * cz2) + 3. / 4. * exp(-1. / 49. * cx1 - 9. / 10. * y - 1. / 10. - 9. / 10. * z - 1. / 10.) + 1. / 2. * exp(-1. / 4. * cx7 - 1. / 4. * cy3 - 1. / 4. * cz5) - 1. / 5. * exp(-cx4 - cy7 - cz5);
			}
		} // namespace

		FrankeProblem::FrankeProblem(const std::string &name)
			: ProblemWithSolution(name)
		{
		}

		VectorNd FrankeProblem::eval_fun(const VectorNd &pt, const double t) const
		{
			VectorNd res(1);

			if (pt.size() == 2)
				res(0) = franke_fun(pt(0), pt(1)) * t;
			else if (pt.size() == 3)
				res(0) = franke_fun(pt(0), pt(1), pt(2)) * t;
			else
				assert(false);

			return res;
		}

		AutodiffGradPt FrankeProblem::eval_fun(const AutodiffGradPt &pt, const double t) const
		{
			AutodiffGradPt res(1);

			if (pt.size() == 2)
				res(0) = franke_fun(pt(0), pt(1)) * t;
			else if (pt.size() == 3)
				res(0) = franke_fun(pt(0), pt(1), pt(2)) * t;
			else
				assert(false);

			return res;
		}

		AutodiffHessianPt FrankeProblem::eval_fun(const AutodiffHessianPt &pt, const double t) const
		{
			AutodiffHessianPt res(1);

			if (pt.size() == 2)
				res(0) = franke_fun(pt(0), pt(1)) * t;
			else if (pt.size() == 3)
				res(0) = franke_fun(pt(0), pt(1), pt(2)) * t;
			else
				assert(false);

			return res;
		}

		////////////////////////////////////////

		FrankeProblemOld::FrankeProblemOld(const std::string &name)
			: ProblemWithSolution(name)
		{
		}

		VectorNd FrankeProblemOld::eval_fun(const VectorNd &pt, const double t) const
		{
			VectorNd res(1);

			if (pt.size() == 2)
				res(0) = franke_fun(pt(0), pt(1)) * t;
			else if (pt.size() == 3)
				res(0) = franke_fun_old(pt(0), pt(1), pt(2)) * t;
			else
				assert(false);

			return res;
		}

		AutodiffGradPt FrankeProblemOld::eval_fun(const AutodiffGradPt &pt, const double t) const
		{
			AutodiffGradPt res(1);

			if (pt.size() == 2)
				res(0) = franke_fun(pt(0), pt(1)) * t;
			else if (pt.size() == 3)
				res(0) = franke_fun_old(pt(0), pt(1), pt(2)) * t;
			else
				assert(false);

			return res;
		}

		AutodiffHessianPt FrankeProblemOld::eval_fun(const AutodiffHessianPt &pt, const double t) const
		{
			AutodiffHessianPt res(1);

			if (pt.size() == 2)
				res(0) = franke_fun(pt(0), pt(1)) * t;
			else if (pt.size() == 3)
				res(0) = franke_fun_old(pt(0), pt(1), pt(2)) * t;
			else
				assert(false);

			return res;
		}
	} // namespace problem
} // namespace polyfem
