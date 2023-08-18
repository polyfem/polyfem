#include "ProblemWithSolution.hpp"

namespace polyfem
{
	namespace problem
	{
		ProblemWithSolution::ProblemWithSolution(const std::string &name)
			: Problem(name)
		{
		}

		void ProblemWithSolution::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			const int size = size_for(pts);
			val.resize(pts.rows(), size);

			for (long i = 0; i < pts.rows(); ++i)
			{
				DiffScalarBase::setVariableCount(pts.cols());
				AutodiffHessianPt pt(pts.cols());

				for (long d = 0; d < pts.cols(); ++d)
					pt(d) = AutodiffScalarHessian(d, pts(i, d));

				const auto res = eval_fun(pt, t);

				val.row(i) = assembler.compute_rhs(res).transpose();
			}
		}

		void ProblemWithSolution::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			exact(pts, t, val);
		}

		void ProblemWithSolution::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val.resize(pts.rows(), size_for(pts));

			for (long i = 0; i < pts.rows(); ++i)
			{
				val.row(i) = eval_fun(VectorNd(pts.row(i)), t);
			}
		}

		void ProblemWithSolution::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			const int size = size_for(pts);
			val.resize(pts.rows(), pts.cols() * size);

			for (long i = 0; i < pts.rows(); ++i)
			{
				DiffScalarBase::setVariableCount(pts.cols());
				AutodiffGradPt pt(pts.cols());

				for (long d = 0; d < pts.cols(); ++d)
					pt(d) = AutodiffScalarGrad(d, pts(i, d));

				const auto res = eval_fun(pt, t);

				for (int m = 0; m < size; ++m)
				{
					const auto &tmp = res(m);
					val.block(i, m * pts.cols(), 1, pts.cols()) = tmp.getGradient().transpose();
				}
			}
		}

		BilaplacianProblemWithSolution::BilaplacianProblemWithSolution(const std::string &name)
			: Problem(name)
		{
		}

		void BilaplacianProblemWithSolution::rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val.resize(pts.rows(), 1);

			for (long i = 0; i < pts.rows(); ++i)
			{
				const double x = pts(i, 0);
				const double y = pts(i, 1);
				val(i) = 24 * x * x * x * x - 48 * x * x * x + 288 * (y - 0.5) * (y - 0.5) * x * x + (-288 * y * y + 288 * y - 48) * x + 24 * y * y * y * y - 48 * y * y * y + 72 * y * y - 48 * y + 8;
			}
		}

		void BilaplacianProblemWithSolution::dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			exact(pts, t, val);
		}

		void BilaplacianProblemWithSolution::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val.resize(pts.rows(), 1);

			for (long i = 0; i < pts.rows(); ++i)
			{
				const double x = pts(i, 0);
				const double y = pts(i, 1);
				val(i) = x * x * (1 - x) * (1 - x) * y * y * (1 - y) * (1 - y);
			}
		}

		void BilaplacianProblemWithSolution::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
		{
			val.resize(pts.rows(), pts.cols());
		}
	} // namespace problem
} // namespace polyfem
