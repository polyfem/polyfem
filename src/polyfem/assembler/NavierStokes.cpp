#include "NavierStokes.hpp"

namespace polyfem::assembler
{

	NavierStokesVelocity::NavierStokesVelocity()
		: viscosity_("viscosity")
	{
	}

	void NavierStokesVelocity::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(codomain_size() == 2 || codomain_size() == 3);

		viscosity_.add_multimaterial(index, params, units.viscosity());
	}

	VectorNd
	NavierStokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == codomain_size());
		VectorNd res(codomain_size());

		VectorNd val(codomain_size());
		for (int d = 0; d < codomain_size(); ++d)
			val(d) = pt(d).getValue();

		const auto nu = viscosity_(val, 0, 0);
		for (int d = 0; d < codomain_size(); ++d)
		{
			res(d) = -val.dot(pt(d).getGradient()) + nu * pt(d).getHessian().trace();
		}

		return res;
	}

	Eigen::VectorXd
	NavierStokesVelocity::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		assert(false);
		return Eigen::VectorXd(data.vals.basis_values.size() * codomain_size());
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
		typedef MatrixNd GradMat;

		assert(data.x.cols() == 1);

		const int n_pts = data.da.size();
		const int n_bases = data.vals.basis_values.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_vel(n_bases * codomain_size(), 1);
		local_vel.setZero();
		for (size_t i = 0; i < n_bases; ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < codomain_size(); ++d)
				{
					local_vel(i * codomain_size() + d) += bs.global[ii].val * data.x(bs.global[ii].index * codomain_size() + d);
				}
			}
		}

		Eigen::MatrixXd N(codomain_size() * n_bases, codomain_size() * n_bases);
		N.setZero();

		GradMat grad_i(codomain_size(), codomain_size());
		GradMat jac_it(codomain_size(), codomain_size());

		Eigen::VectorXd vel(codomain_size(), 1);
		VectorNd phi_j(codomain_size(), 1);

		for (long p = 0; p < n_pts; ++p)
		{
			vel.setZero();

			for (size_t i = 0; i < n_bases; ++i)
			{
				const auto &bs = data.vals.basis_values[i];
				const double val = bs.val(p);

				for (int d = 0; d < codomain_size(); ++d)
				{
					vel(d) += val * local_vel(i * codomain_size() + d);
				}
			}

			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = data.vals.jac_it[p](k);

			for (int i = 0; i < n_bases; ++i)
			{
				const auto &bi = data.vals.basis_values[i];
				for (int m = 0; m < codomain_size(); ++m)
				{
					grad_i.setZero();
					grad_i.row(m) = bi.grad.row(p);
					grad_i = grad_i * jac_it;

					for (int j = 0; j < n_bases; ++j)
					{
						const auto &bj = data.vals.basis_values[j];
						for (int n = 0; n < codomain_size(); ++n)
						{
							phi_j.setZero();
							phi_j(n) = bj.val(p);
							N(i * codomain_size() + m, j * codomain_size() + n) += (grad_i * vel).dot(phi_j) * data.da(p);
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
		typedef MatrixNd GradMat;

		assert(data.x.cols() == 1);

		const int n_pts = data.da.size();
		const int n_bases = data.vals.basis_values.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_vel(n_bases * codomain_size(), 1);
		local_vel.setZero();
		for (size_t i = 0; i < n_bases; ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < codomain_size(); ++d)
				{
					local_vel(i * codomain_size() + d) += bs.global[ii].val * data.x(bs.global[ii].index * codomain_size() + d);
				}
			}
		}

		Eigen::MatrixXd W(codomain_size() * n_bases, codomain_size() * n_bases);
		W.setZero();

		GradMat grad_v(codomain_size(), codomain_size());
		GradMat jac_it(codomain_size(), codomain_size());

		VectorNd phi_i(codomain_size(), 1);
		VectorNd phi_j(codomain_size(), 1);

		for (long p = 0; p < n_pts; ++p)
		{
			grad_v.setZero();

			for (size_t i = 0; i < n_bases; ++i)
			{
				const auto &bs = data.vals.basis_values[i];
				const VectorNd grad = bs.grad.row(p);
				assert(grad.size() == codomain_size());

				for (int d = 0; d < codomain_size(); ++d)
				{
					for (int c = 0; c < codomain_size(); ++c)
					{
						grad_v(d, c) += grad(c) * local_vel(i * codomain_size() + d);
					}
				}
			}

			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = data.vals.jac_it[p](k);
			grad_v = (grad_v * jac_it).eval();

			for (int i = 0; i < n_bases; ++i)
			{
				const auto &bi = data.vals.basis_values[i];
				for (int m = 0; m < codomain_size(); ++m)
				{
					phi_i.setZero();
					phi_i(m) = bi.val(p);

					for (int j = 0; j < n_bases; ++j)
					{
						const auto &bj = data.vals.basis_values[j];
						for (int n = 0; n < codomain_size(); ++n)
						{
							phi_j.setZero();
							phi_j(n) = bj.val(p);
							W(i * codomain_size() + m, j * codomain_size() + n) += (grad_v * phi_i).dot(phi_j) * data.da(p);
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
		const auto &nu = viscosity_;
		res["viscosity"] = [&nu](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return nu(p, t, e);
		};

		return res;
		return res;
	}
} // namespace polyfem::assembler
