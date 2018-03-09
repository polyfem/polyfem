#include "FrankeProblem.hpp"

#include <iostream>

namespace poly_fem
{
	namespace
	{
		template<typename T>
		T franke_fun(T x, T y)
		{
			auto cx2 = (9*x-2) * (9*x-2);
			auto cy2 = (9*y-2) * (9*y-2);

			auto cx1 = (9*x+1) * (9*x+1);
			auto cx7 = (9*x-7) * (9*x-7);

			auto cy3 = (9*y-3) * (9*y-3);
			auto cx4 = (9*x-4) * (9*x-4);

			auto cy7 = (9*y-7) * (9*y-7);

			return (3./4.)*exp(-(1./4.)*cx2-(1./4.)*cy2)+(3./4.)*exp(-(1./49.)*cx1-(9./10.)*y-1./10.)+(1./2.)*exp(-(1./4.)*cx7-(1./4.)*cy3)-(1./5.)*exp(-cx4-cy7);
		}


		template<typename T>
		T franke_fun(T x, T y, T z)
		{
			auto cx2 = (9*x-2) * (9*x-2);
			auto cy2 = (9*y-2) * (9*y-2);
			auto cz2 = (9*z-2) * (9*z-2);

			auto cx1 = (9*x+1) * (9*x+1);
			auto cx7 = (9*x-7) * (9*x-7);

			auto cy3 = (9*y-3) * (9*y-3);
			auto cx4 = (9*x-4) * (9*x-4);
			auto cy7 = (9*y-7) * (9*y-7);

			auto cz5 = (9*y-5) * (9*y-5);

			return
			3./4. * exp( -1./4.*cx2 - 1./4.*cy2 - 1./4.*cz2) +
			3./4. * exp(-1./49. * cx1 - 9./10.*y - 1./10. -  9./10.*z - 1./10.) +
			1./2. * exp(-1./4. * cx7 - 1./4. * cy3 - 1./4. * cz5) -
			1./5. * exp(- cx4 - cy7 - cz5);
		}
	}



	Franke2dProblem::Franke2dProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }


	VectorNd Franke2dProblem::eval_fun(const VectorNd &pt) const
	{
		assert(pt.size() == 2);

		VectorNd res(1);
		res(0) = franke_fun(pt(0), pt(1));
		return res;
	}

	AutodiffGradPt Franke2dProblem::eval_fun(const AutodiffGradPt &pt) const
	{
		assert(pt.size() == 2);

		AutodiffGradPt res(1);
		res(0) = franke_fun(pt(0), pt(1));
		return res;
	}

	AutodiffHessianPt Franke2dProblem::eval_fun(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == 2);

		AutodiffHessianPt res(1);
		res(0) = franke_fun(pt(0), pt(1));
		return res;
	}







	Franke3dProblem::Franke3dProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd Franke3dProblem::eval_fun(const VectorNd &pt) const
	{
		assert(pt.size() == 3);

		VectorNd res(1);
		res(0) = franke_fun(pt(0), pt(1), pt(2));
		return res;
	}

	AutodiffGradPt Franke3dProblem::eval_fun(const AutodiffGradPt &pt) const
	{
		assert(pt.size() == 3);

		AutodiffGradPt res(1);
		res(0) = franke_fun(pt(0), pt(1), pt(2));
		return res;
	}

	AutodiffHessianPt Franke3dProblem::eval_fun(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == 3);

		AutodiffHessianPt res(1);
		res(0) = franke_fun(pt(0), pt(1), pt(2));
		return res;
	}

}
