#include "NavierStokes.hpp"

namespace polyfem::assembler
{

	void NavierStokesVelocity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);
		// TODO use units and generic mat param

		if (params.count("viscosity"))
		{
			viscosity_ = params["viscosity"];
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	NavierStokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> val(size());
		for (int d = 0; d < size(); ++d)
			val(d) = pt(d).getValue();

		for (int d = 0; d < size(); ++d)
		{
			res(d) = -val.dot(pt(d).getGradient()) + viscosity_ * pt(d).getHessian().trace();
		}

		return res;
	}

	Eigen::VectorXd
	NavierStokesVelocity::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		assert(false);
		return Eigen::VectorXd(data.vals.basis_values.size() * size());
	}

	Eigen::MatrixXd
	NavierStokesVelocity::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		Eigen::MatrixXd H;
		if (full_gradient_)
			H = compute_N(data) + compute_W(data);
		else
			H = compute_N(data);

		return H.transpose();
	}

	// Compute N = int v \cdot \nabla phi_i \cdot \phi_j

	Eigen::MatrixXd NavierStokesVelocity::compute_N(const NonLinearAssemblerData &data) const
	{
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> GradMat;

		assert(data.x.cols() == 1);

		const int n_pts = data.da.size();
		const int n_bases = data.vals.basis_values.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_vel(n_bases * size(), 1);
		local_vel.setZero();
		for (size_t i = 0; i < n_bases; ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_vel(i * size() + d) += bs.global[ii].val * data.x(bs.global[ii].index * size() + d);
				}
			}
		}

		Eigen::MatrixXd N(size() * n_bases, size() * n_bases);
		N.setZero();

		GradMat grad_i(size(), size());
		GradMat jac_it(size(), size());

		Eigen::VectorXd vel(size(), 1);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> phi_j(size(), 1);

		for (long p = 0; p < n_pts; ++p)
		{
			vel.setZero();

			for (size_t i = 0; i < n_bases; ++i)
			{
				const auto &bs = data.vals.basis_values[i];
				const double val = bs.val(p);

				for (int d = 0; d < size(); ++d)
				{
					vel(d) += val * local_vel(i * size() + d);
				}
			}

			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = data.vals.jac_it[p](k);

			for (int i = 0; i < n_bases; ++i)
			{
				const auto &bi = data.vals.basis_values[i];
				for (int m = 0; m < size(); ++m)
				{
					grad_i.setZero();
					grad_i.row(m) = bi.grad.row(p);
					grad_i = grad_i * jac_it;

					for (int j = 0; j < n_bases; ++j)
					{
						const auto &bj = data.vals.basis_values[j];
						for (int n = 0; n < size(); ++n)
						{
							phi_j.setZero();
							phi_j(n) = bj.val(p);
							N(i * size() + m, j * size() + n) += (grad_i * vel).dot(phi_j) * data.da(p);
						}
					}
				}
			}
		}

		return N;
	}

	// Compute N = int phi_j \cdot \nabla v \cdot \phi_j

	Eigen::MatrixXd NavierStokesVelocity::compute_W(const NonLinearAssemblerData &data) const
	{
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> GradMat;

		assert(data.x.cols() == 1);

		const int n_pts = data.da.size();
		const int n_bases = data.vals.basis_values.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_vel(n_bases * size(), 1);
		local_vel.setZero();
		for (size_t i = 0; i < n_bases; ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_vel(i * size() + d) += bs.global[ii].val * data.x(bs.global[ii].index * size() + d);
				}
			}
		}

		Eigen::MatrixXd W(size() * n_bases, size() * n_bases);
		W.setZero();

		GradMat grad_v(size(), size());
		GradMat jac_it(size(), size());

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> phi_i(size(), 1);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> phi_j(size(), 1);

		for (long p = 0; p < n_pts; ++p)
		{
			grad_v.setZero();

			for (size_t i = 0; i < n_bases; ++i)
			{
				const auto &bs = data.vals.basis_values[i];
				const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad = bs.grad.row(p);
				assert(grad.size() == size());

				for (int d = 0; d < size(); ++d)
				{
					for (int c = 0; c < size(); ++c)
					{
						grad_v(d, c) += grad(c) * local_vel(i * size() + d);
					}
				}
			}

			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = data.vals.jac_it[p](k);
			grad_v = (grad_v * jac_it).eval();

			for (int i = 0; i < n_bases; ++i)
			{
				const auto &bi = data.vals.basis_values[i];
				for (int m = 0; m < size(); ++m)
				{
					phi_i.setZero();
					phi_i(m) = bi.val(p);

					for (int j = 0; j < n_bases; ++j)
					{
						const auto &bj = data.vals.basis_values[j];
						for (int n = 0; n < size(); ++n)
						{
							phi_j.setZero();
							phi_j(n) = bj.val(p);
							W(i * size() + m, j * size() + n) += (grad_v * phi_i).dot(phi_j) * data.da(p);
						}
					}
				}
			}
		}

		return W;
	}

	std::map<std::string, Assembler::ParamFunc> NavierStokesVelocity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		res["k"] = [this](const RowVectorNd &, const RowVectorNd &, double, int) { return this->viscosity_; };

		return res;
	}
} // namespace polyfem::assembler
