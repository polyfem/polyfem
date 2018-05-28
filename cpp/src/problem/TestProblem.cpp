#include "TestProblem.hpp"

#include <iostream>

namespace poly_fem
{
	namespace
	{
		template<typename T>
		T reentrant_corner(double omega, T x, T y)
		{
			const double alpha = M_PI/omega;
			const T r = sqrt(x*x+y*y);
			const T theta = atan2(y, x);
			return pow(r, alpha)*sin(alpha*theta);
		}
	}

	ReentrantCornerProblem::ReentrantCornerProblem(const std::string &name)
	: ProblemWithSolution(name), omega_(7.*M_PI/4.)
	{ }


	VectorNd ReentrantCornerProblem::eval_fun(const VectorNd &pt) const
	{
		VectorNd res(1);
		res(0) = reentrant_corner(omega_, pt(0), pt(1));

		return res;
	}

	void ReentrantCornerProblem::set_parameters(const json &params)
	{
		if(params.count("omega"))
			omega_ = params["omega"];
	}

	AutodiffGradPt ReentrantCornerProblem::eval_fun(const AutodiffGradPt &pt) const
	{
		AutodiffGradPt res(1);
		res(0) = reentrant_corner(omega_, pt(0), pt(1));

		return res;
	}

	AutodiffHessianPt ReentrantCornerProblem::eval_fun(const AutodiffHessianPt &pt) const
	{
		AutodiffHessianPt res(1);
		res(0) = reentrant_corner(omega_, pt(0), pt(1));

		return res;
	}


}
