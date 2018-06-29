#include <polyfem/ProblemWithSolution.hpp>


#include <polyfem/AssemblerUtils.hpp>

namespace poly_fem
{
	ProblemWithSolution::ProblemWithSolution(const std::string &name)
	: Problem(name)
	{ }

	void ProblemWithSolution::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		const auto &assembler = AssemblerUtils::instance();

		const int size = size_for(pts);
		val.resize(pts.rows(), size);


		for(long i = 0; i < pts.rows(); ++i)
		{
			DiffScalarBase::setVariableCount(pts.cols());
			AutodiffHessianPt pt(pts.cols());

			for(long d = 0; d < pts.cols(); ++d)
				pt(d) = AutodiffScalarHessian(d, pts(i, d));

			const auto res = eval_fun(pt);

			val.row(i) = assembler.compute_rhs(formulation, res).transpose();
		}

		val *= t;
	}

	void ProblemWithSolution::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
		val *= t;
	}

	void ProblemWithSolution::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		val.resize(pts.rows(), size_for(pts));

		for(long i = 0; i < pts.rows(); ++i)
		{
			val.row(i) = eval_fun(VectorNd(pts.row(i)));
		}
	}

	void ProblemWithSolution::exact_grad(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		const int size = size_for(pts);
		val.resize(pts.rows(), pts.cols() * size);

		for(long i = 0; i < pts.rows(); ++i)
		{
			DiffScalarBase::setVariableCount(pts.cols());
			AutodiffGradPt pt(pts.cols());

			for(long d = 0; d < pts.cols(); ++d)
				pt(d) = AutodiffScalarGrad(d, pts(i, d));

			const auto res = eval_fun(pt);

			for(int m = 0; m < size; ++m)
			{
				const auto &tmp = res(m);
				val.block(i, m*pts.cols(), 1, pts.cols()) = tmp.getGradient().transpose();
			}
		}
	}


}
