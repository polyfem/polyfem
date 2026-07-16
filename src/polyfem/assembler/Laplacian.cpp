#include "Laplacian.hpp"

namespace polyfem::assembler
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}

		Eigen::VectorXd local_scalar_values(const NonLinearAssemblerData &data)
		{
			assert(data.x.cols() == 1);

			const int n_bases = int(data.vals.basis_values.size());
			Eigen::VectorXd local_u = Eigen::VectorXd::Zero(n_bases);
			for (int i = 0; i < n_bases; ++i)
			{
				const auto &bs = data.vals.basis_values[i];
				for (const auto &global : bs.global)
					local_u(i) += global.val * data.x(global.index);
			}

			return local_u;
		}
	} // namespace

	Laplacian::Laplacian(const std::string &conductivity_param_name)
		: conductivity_param_name_(conductivity_param_name),
		  conductivity_(conductivity_param_name.empty() ? "conductivity" : conductivity_param_name)
	{
	}

	std::map<std::string, Assembler::ParamFunc> Laplacian::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		if (!conductivity_param_name_.empty())
		{
			res[conductivity_param_name_] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
				return conductivity(uv, p, t, e);
			};
		}

		return res;
	}

	void Laplacian::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		if (!conductivity_param_name_.empty())
			conductivity_.add_multimaterial(index, params, units.thermal_conductivity(), root_path);
	}

	double Laplacian::conductivity(const RowVectorNd &, const RowVectorNd &p, double t, int element_id) const
	{
		if (conductivity_param_name_.empty())
			return 1.0;

		return conductivity_(p, t, element_id);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> Laplacian::assemble(const LinearAssemblerData &data) const
	{
		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;
		// return ((gradi.array() * gradj.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		assert(gradi.rows() == data.da.size());
		for (int k = 0; k < gradi.rows(); ++k)
		{
			const double kappa = conductivity(data.vals.quadrature.points.row(k), data.vals.val.row(k), data.t, data.vals.element_id);
			// compute grad(phi_i) dot grad(phi_j) weighted by quadrature weights
			res += kappa * gradi.row(k).dot(gradj.row(k)) * data.da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	double Laplacian::compute_energy(const NonLinearAssemblerData &data) const
	{
		assert(size() == 1);

		const Eigen::VectorXd local_u = local_scalar_values(data);
		return 0.5 * local_u.dot(assemble_hessian(data) * local_u);
	}

	Eigen::VectorXd Laplacian::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		assert(size() == 1);

		return assemble_hessian(data) * local_scalar_values(data);
	}

	Eigen::MatrixXd Laplacian::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		assert(size() == 1);

		const int n_bases = int(data.vals.basis_values.size());
		Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(n_bases, n_bases);
		for (int i = 0; i < n_bases; ++i)
		{
			for (int j = 0; j < n_bases; ++j)
				hessian(i, j) = assemble(LinearAssemblerData(data.vals, data.t, i, j, data.da))(0);
		}

		return hessian;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> Laplacian::compute_rhs(const AutodiffHessianPt &pt) const
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
			const double kappa = conductivity(vals.quadrature.points.row(p), vals.val.row(p), t, vals.element_id);
			for (int i = 0, idx = 0; i < dim; i++)
				for (int j = 0; j < dim; j++)
				{
					tensor(p, idx) = kappa * delta(i, j);
					idx++;
				}
		}
	}
} // namespace polyfem::assembler
