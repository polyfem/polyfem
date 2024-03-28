#include "Laplacian.hpp"

namespace polyfem::assembler
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}
	} // namespace

	FlatMatrixNd Laplacian::assemble(const LinearAssemblerData &data) const
	{
		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;
		// return ((gradi.array() * gradj.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		assert(gradi.rows() == data.da.size());
		for (int k = 0; k < gradi.rows(); ++k)
		{
			// compute grad(phi_i) dot grad(phi_j) weighted by quadrature weights
			res += gradi.row(k).dot(gradj.row(k)) * data.da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	VectorNd Laplacian::compute_rhs(const AutodiffHessianPt &pt) const
	{
		Eigen::Matrix<double, 1, 1> result;
		assert(pt.size() == 1);
		result(0) = pt(0).getHessian().trace();
		return result;
	}

	Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> Laplacian::kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const
	{
		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(1);

		if (dim == 2)
			res(0) = -1. / (2 * M_PI) * log(r);
		else if (dim == 3)
			res(0) = 1. / (4 * M_PI * r);
		else
			assert(false);

		return res;
	}

	void Laplacian::compute_stress_grad_multiply_mat(const OptAssemblerData &data,
													 const Eigen::MatrixXd &mat,
													 Eigen::MatrixXd &stress,
													 Eigen::MatrixXd &result) const
	{
		stress = data.grad_u_i;
		result = mat;
	}

	void Laplacian::compute_stiffness_value(const double t,
											const assembler::ElementAssemblyValues &vals,
											const Eigen::MatrixXd &local_pts,
											const Eigen::MatrixXd &displacement,
											Eigen::MatrixXd &tensor) const
	{
		const int dim = local_pts.cols();
		tensor.resize(local_pts.rows(), dim * dim);
		assert(displacement.cols() == 1);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			for (int i = 0, idx = 0; i < dim; i++)
				for (int j = 0; j < dim; j++)
				{
					tensor(p, idx) = delta(i, j);
					idx++;
				}
		}
	}
} // namespace polyfem::assembler
