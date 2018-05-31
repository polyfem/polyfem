#include "MiscProblem.hpp"

#include <iostream>

namespace poly_fem
{
	namespace
	{
		template<typename T>
		T linear_fun(T x, T y)
		{
			return x;
		}

		template<typename T>
		T quadratic_fun(T x, T y)
		{
			T v = x;
			return v*v;
		}

		template<typename T>
		T cubic_fun(T x, T y)
		{
			T v = (2*y-0.9);
			return v*v*v*v + 0.1;
		}

		template<typename T>
		T sine_fun(T x, T y)
		{
			return sin(x*10)*sin(y*10);
		}

		template<typename T>
		T sine_fun(T x, T y, T z)
		{
			return sin(x*10)*sin(y*10)*sin(z*10);
		}

		template<typename T>
		T zero_bc(T x, T y)
		{
			return (1 - x)  * x * x * y * (1-y) *(1-y);
		}

		template<typename T>
		T zero_bc(T x, T y, T z)
		{
			return (1 - x)  * x * x * y * (1-y) *(1-y) * z * (1 - z);
		}
	}

	LinearProblem::LinearProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }


	VectorNd LinearProblem::eval_fun(const VectorNd &pt) const
	{
		VectorNd res(1);
		res(0) = linear_fun(pt(0), pt(1));

		return res;
	}

	AutodiffGradPt LinearProblem::eval_fun(const AutodiffGradPt &pt) const
	{
		AutodiffGradPt res(1);
		res(0) = linear_fun(pt(0), pt(1));

		return res;
	}

	AutodiffHessianPt LinearProblem::eval_fun(const AutodiffHessianPt &pt) const
	{
		AutodiffHessianPt res(1);
		res(0) = linear_fun(pt(0), pt(1));

		return res;
	}







	QuadraticProblem::QuadraticProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd QuadraticProblem::eval_fun(const VectorNd &pt) const
	{
		VectorNd res(1);
		res(0) = quadratic_fun(pt(0), pt(1));

		return res;
	}

	AutodiffGradPt QuadraticProblem::eval_fun(const AutodiffGradPt &pt) const
	{
		AutodiffGradPt res(1);
		res(0) = quadratic_fun(pt(0), pt(1));

		return res;
	}

	AutodiffHessianPt QuadraticProblem::eval_fun(const AutodiffHessianPt &pt) const
	{
		AutodiffHessianPt res(1);
		res(0) = quadratic_fun(pt(0), pt(1));

		return res;
	}



	CubicProblem::CubicProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd CubicProblem::eval_fun(const VectorNd &pt) const
	{
		VectorNd res(1);
		res(0) = cubic_fun(pt(0), pt(1));

		return res;
	}

	AutodiffGradPt CubicProblem::eval_fun(const AutodiffGradPt &pt) const
	{
		AutodiffGradPt res(1);
		res(0) = cubic_fun(pt(0), pt(1));

		return res;
	}

	AutodiffHessianPt CubicProblem::eval_fun(const AutodiffHessianPt &pt) const
	{
		AutodiffHessianPt res(1);
		res(0) = cubic_fun(pt(0), pt(1));

		return res;
	}


	SineProblem::SineProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd SineProblem::eval_fun(const VectorNd &pt) const
	{
		VectorNd res(1);
		if(pt.size() == 2)
			res(0) = sine_fun(pt(0), pt(1));
		else if(pt.size() == 3)
			res(0) = sine_fun(pt(0), pt(1), pt(2));
		else
			assert(false);

		return res;
	}

	AutodiffGradPt SineProblem::eval_fun(const AutodiffGradPt &pt) const
	{
		AutodiffGradPt res(1);

		if(pt.size() == 2)
			res(0) = sine_fun(pt(0), pt(1));
		else if(pt.size() == 3)
			res(0) = sine_fun(pt(0), pt(1), pt(2));
		else
			assert(false);

		return res;
	}

	AutodiffHessianPt SineProblem::eval_fun(const AutodiffHessianPt &pt) const
	{
		AutodiffHessianPt res(1);

		if(pt.size() == 2)
			res(0) = sine_fun(pt(0), pt(1));
		else if(pt.size() == 3)
			res(0) = sine_fun(pt(0), pt(1), pt(2));
		else
			assert(false);

		return res;
	}



	ZeroBCProblem::ZeroBCProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd ZeroBCProblem::eval_fun(const VectorNd &pt) const
	{
		VectorNd res(1);
		if(pt.size() == 2)
			res(0) = zero_bc(pt(0), pt(1));
		else if(pt.size() == 3)
			res(0) = zero_bc(pt(0), pt(1), pt(2));
		else
			assert(false);

		return res;
	}

	AutodiffGradPt ZeroBCProblem::eval_fun(const AutodiffGradPt &pt) const
	{
		AutodiffGradPt res(1);
		if(pt.size() == 2)
			res(0) = zero_bc(pt(0), pt(1));
		else if(pt.size() == 3)
			res(0) = zero_bc(pt(0), pt(1), pt(2));
		else
			assert(false);

		return res;
	}

	AutodiffHessianPt ZeroBCProblem::eval_fun(const AutodiffHessianPt &pt) const
	{
		AutodiffHessianPt res(1);
		if(pt.size() == 2)
			res(0) = zero_bc(pt(0), pt(1));
		else if(pt.size() == 3)
			res(0) = zero_bc(pt(0), pt(1), pt(2));
		else
			assert(false);

		return res;
	}

}
