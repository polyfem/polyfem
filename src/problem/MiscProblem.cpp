#include <polyfem/MiscProblem.hpp>

#include <iostream>

namespace polyfem
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


	VectorNd LinearProblem::eval_fun(const VectorNd &pt, const double t) const
	{
		VectorNd res(1);
		res(0) = linear_fun(pt(0), pt(1)) * t;

		return res;
	}

	AutodiffGradPt LinearProblem::eval_fun(const AutodiffGradPt &pt, const double t) const
	{
		AutodiffGradPt res(1);
		res(0) = linear_fun(pt(0), pt(1)) * t;

		return res;
	}

	AutodiffHessianPt LinearProblem::eval_fun(const AutodiffHessianPt &pt, const double t) const
	{
		AutodiffHessianPt res(1);
		res(0) = linear_fun(pt(0), pt(1)) * t;

		return res;
	}







	QuadraticProblem::QuadraticProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd QuadraticProblem::eval_fun(const VectorNd &pt, const double t) const
	{
		VectorNd res(1);
		res(0) = quadratic_fun(pt(0), pt(1)) * t;

		return res;
	}

	AutodiffGradPt QuadraticProblem::eval_fun(const AutodiffGradPt &pt, const double t) const
	{
		AutodiffGradPt res(1);
		res(0) = quadratic_fun(pt(0), pt(1)) * t;

		return res;
	}

	AutodiffHessianPt QuadraticProblem::eval_fun(const AutodiffHessianPt &pt, const double t) const
	{
		AutodiffHessianPt res(1);
		res(0) = quadratic_fun(pt(0), pt(1)) * t;

		return res;
	}



	CubicProblem::CubicProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd CubicProblem::eval_fun(const VectorNd &pt, const double t) const
	{
		VectorNd res(1);
		res(0) = cubic_fun(pt(0), pt(1)) * t;

		return res;
	}

	AutodiffGradPt CubicProblem::eval_fun(const AutodiffGradPt &pt, const double t) const
	{
		AutodiffGradPt res(1);
		res(0) = cubic_fun(pt(0), pt(1)) * t;

		return res;
	}

	AutodiffHessianPt CubicProblem::eval_fun(const AutodiffHessianPt &pt, const double t) const
	{
		AutodiffHessianPt res(1);
		res(0) = cubic_fun(pt(0), pt(1)) * t;

		return res;
	}


	SineProblem::SineProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd SineProblem::eval_fun(const VectorNd &pt, const double t) const
	{
		VectorNd res(1);
		if(pt.size() == 2)
			res(0) = sine_fun(pt(0), pt(1)) * t;
		else if(pt.size() == 3)
			res(0) = sine_fun(pt(0), pt(1), pt(2)) * t;
		else
			assert(false);

		return res;
	}

	AutodiffGradPt SineProblem::eval_fun(const AutodiffGradPt &pt, const double t) const
	{
		AutodiffGradPt res(1);

		if(pt.size() == 2)
			res(0) = sine_fun(pt(0), pt(1)) * t;
		else if(pt.size() == 3)
			res(0) = sine_fun(pt(0), pt(1), pt(2)) * t;
		else
			assert(false);

		return res;
	}

	AutodiffHessianPt SineProblem::eval_fun(const AutodiffHessianPt &pt, const double t) const
	{
		AutodiffHessianPt res(1);

		if(pt.size() == 2)
			res(0) = sine_fun(pt(0), pt(1)) * t;
		else if(pt.size() == 3)
			res(0) = sine_fun(pt(0), pt(1), pt(2)) * t;
		else
			assert(false);

		return res;
	}



	ZeroBCProblem::ZeroBCProblem(const std::string &name)
	: ProblemWithSolution(name)
	{ }

	VectorNd ZeroBCProblem::eval_fun(const VectorNd &pt, const double t) const
	{
		VectorNd res(1);
		if(pt.size() == 2)
			res(0) = zero_bc(pt(0), pt(1)) * t;
		else if(pt.size() == 3)
			res(0) = zero_bc(pt(0), pt(1), pt(2)) * t;
		else
			assert(false);

		return res;
	}

	AutodiffGradPt ZeroBCProblem::eval_fun(const AutodiffGradPt &pt, const double t) const
	{
		AutodiffGradPt res(1);
		if(pt.size() == 2)
			res(0) = zero_bc(pt(0), pt(1)) * t;
		else if(pt.size() == 3)
			res(0) = zero_bc(pt(0), pt(1), pt(2)) * t;
		else
			assert(false);

		return res;
	}

	AutodiffHessianPt ZeroBCProblem::eval_fun(const AutodiffHessianPt &pt, const double t) const
	{
		AutodiffHessianPt res(1);
		if(pt.size() == 2)
			res(0) = zero_bc(pt(0), pt(1)) * t;
		else if(pt.size() == 3)
			res(0) = zero_bc(pt(0), pt(1), pt(2)) * t;
		else
			assert(false);

		return res;
	}

	MinSurfProblem::MinSurfProblem(const std::string &name)
	: Problem(name)
	{ }

	void MinSurfProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const
	{
		val = -10*Eigen::MatrixXd::Ones(pts.rows(), 1);
	}

	void MinSurfProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);
	}

	TimeDependentProblem::TimeDependentProblem(const std::string &name)
	: Problem(name)
	{ }

	void TimeDependentProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Ones(pts.rows(), 1);
	}

	void TimeDependentProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);
	}

	void TimeDependentProblem::initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), 1);
	}

	GenericScalarProblemExact::GenericScalarProblemExact(const std::string &name)
		: ProblemWithSolution(name), func_(0)
	{
	}

	void GenericScalarProblemExact::initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, 0, val);
	}

	void GenericScalarProblemExact::set_parameters(const json &params)
	{

		if (params.find("func") != params.end())
		{
			func_ = params["func"];
		}
	}

	VectorNd GenericScalarProblemExact::eval_fun(const VectorNd &pt, double t) const
	{
		return VectorNd(1);
	}
	AutodiffGradPt GenericScalarProblemExact::eval_fun(const AutodiffGradPt &pt, double t) const
	{
		return AutodiffGradPt(1);
	}
	AutodiffHessianPt GenericScalarProblemExact::eval_fun(const AutodiffHessianPt &pt, double t) const
	{
		return AutodiffHessianPt(1);
	}

	// void GenericScalarProblemExact::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	// {
	// 	val.resize(pts.rows(), pts.cols());
	// 	for (int i = 0; i < pts.rows(); ++i)
	// 	{
	// 		const double x = pts(i, 0);
	// 		const double y = pts(i, 1);

	// 		val(i, 0) = -t + x * x / 2 + x * y;
	// 		val(i, 1) = t - x * y - y * y / 2;
	// 	}
	// }

	// void GenericScalarProblemExact::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	// {
	// 	const double time_scaling = exp(-2 * viscosity_ * t);

	// 	val.resize(pts.rows(), pts.cols() * pts.cols());

	// 	// for (int i = 0; i < pts.rows(); ++i)
	// 	// {
	// 	// 	const double x = pts(i, 0);
	// 	// 	const double y = pts(i, 1);

	// 	// 	val(i, 0) = -sin(x) * sin(y) * time_scaling;
	// 	// 	val(i, 1) = cos(x) * cos(y) * time_scaling;
	// 	// 	val(i, 2) = -cos(x) * cos(y) * time_scaling;
	// 	// 	val(i, 3) = sin(x) * sin(y) * time_scaling;
	// 	// }
	// }

	// void GenericScalarProblemExact::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	// {
	// 	val.resize(pts.rows(), pts.cols());

	// 	for (int i = 0; i < pts.rows(); ++i)
	// 	{
	// 		const double x = pts(i, 0);
	// 		const double y = pts(i, 1);

	// 		val(i, 0) = -viscosity_ - t * y + 1. / 2. * x * (x * x + x * y + y * y);
	// 		val(i, 1) = viscosity_ - t * x + 1. / 2. * y * (x * x + x * y + y * y) + 2;
	// 	}
	// }

	// void GenericScalarProblemExact::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	// {
	// 	exact(pts, t, val);
	// }
}
