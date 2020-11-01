#include <polyfem/NavierStokes.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElasticityUtils.hpp>

namespace polyfem
{
	template<bool full_gradient>
	void NavierStokesVelocity<full_gradient>::set_parameters(const json &params)
	{
		set_size(params["size"]);

		if (params.count("viscosity")) {
			viscosity_ = params["viscosity"];
		}
	}

	template <bool full_gradient>
	void NavierStokesVelocity<full_gradient>::set_size(const int size)
	{
		size_ = size;
	}

	template <bool full_gradient>
	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	NavierStokesVelocity<full_gradient>::compute_rhs(const AutodiffHessianPt &pt) const
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

	template <bool full_gradient>
	void NavierStokesVelocity<full_gradient>::compute_norm_velocity(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const
	{
		norms.resize(local_pts.rows(), 1);
		assert(velocity.cols() == 1);

		Eigen::MatrixXd vel(1, size());

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, gbs);

		for(long p = 0; p < local_pts.rows(); ++p)
		{
			vel.setZero();

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.val.rows() == local_pts.rows());
				assert(loc_val.val.cols() == 1);

				for(int d = 0; d < size(); ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						vel(d) += b.global()[ii].val * loc_val.val(p) * velocity(b.global()[ii].index*size() + d);
					}
				}
			}

			norms(p) = vel.norm();
		}
	}

	template <bool full_gradient>
	void NavierStokesVelocity<full_gradient>::compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const
	{
		tensor.resize(local_pts.rows(), size());
		assert(velocity.cols() == 1);

		Eigen::MatrixXd vel(1, size());

		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, gbs);

		for(long p = 0; p < local_pts.rows(); ++p)
		{
			vel.setZero();

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.grad.rows() == local_pts.rows());
				assert(loc_val.grad.cols() == size());

				for(int d = 0; d < size(); ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						vel(d) += b.global()[ii].val * loc_val.val(p) * velocity(b.global()[ii].index*size() + d);
					}
				}
			}

			tensor.row(p) = vel;
		}
	}

	template <bool full_gradient>
	Eigen::VectorXd
	NavierStokesVelocity<full_gradient>::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
	{
		assert(false);
		return Eigen::VectorXd(vals.basis_values.size()*size());
	}

	template <bool full_gradient>
	Eigen::MatrixXd
	NavierStokesVelocity<full_gradient>::assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
	{
		Eigen::MatrixXd H;
		if (full_gradient)
			H = compute_N(vals, velocity, da) + compute_W(vals, velocity, da);
		else
			H = compute_N(vals, velocity, da);

		return H.transpose();
	}

	//Compute N = int v \cdot \nabla phi_i \cdot \phi_j
	template <bool full_gradient>
	Eigen::MatrixXd NavierStokesVelocity<full_gradient>::compute_N(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
	{
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> GradMat;

		assert(velocity.cols() == 1);

		const int n_pts = da.size();
		const int n_bases = vals.basis_values.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_vel(n_bases * size(), 1);
		local_vel.setZero();
		for (size_t i = 0; i < n_bases; ++i)
		{
			const auto &bs = vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_vel(i * size() + d) += bs.global[ii].val * velocity(bs.global[ii].index * size() + d);
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
			vel .setZero();

			for (size_t i = 0; i < n_bases; ++i)
			{
				const auto &bs = vals.basis_values[i];
				const double val = bs.val(p);

				for (int d = 0; d < size(); ++d)
				{
					vel(d) += val * local_vel(i * size() + d);
				}
			}

			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = vals.jac_it[p](k);

			for (int i = 0; i < n_bases; ++i)
			{
				const auto &bi = vals.basis_values[i];
				for (int m = 0; m < size(); ++m)
				{
					grad_i.setZero();
					grad_i.row(m) = bi.grad.row(p);
					grad_i = grad_i * jac_it;

					for (int j = 0; j < n_bases; ++j)
					{
						const auto &bj = vals.basis_values[j];
						for (int n = 0; n < size(); ++n)
						{
							phi_j.setZero();
							phi_j(n) = bj.val(p);
							N(i * size() + m, j * size() + n) += (grad_i * vel).dot(phi_j) * da(p);
						}
					}
				}
			}
		}

		return N;
	}

	//Compute N = int phi_j \cdot \nabla v \cdot \phi_j
	template <bool full_gradient>
	Eigen::MatrixXd NavierStokesVelocity<full_gradient>::compute_W(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
	{
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> GradMat;

		assert(velocity.cols() == 1);

		const int n_pts = da.size();
		const int n_bases = vals.basis_values.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_vel(n_bases * size(), 1);
		local_vel.setZero();
		for (size_t i = 0; i < n_bases; ++i)
		{
			const auto &bs = vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_vel(i * size() + d) += bs.global[ii].val * velocity(bs.global[ii].index * size() + d);
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
				const auto &bs = vals.basis_values[i];
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
				jac_it(k) = vals.jac_it[p](k);
			grad_v = (grad_v * jac_it).eval();

			for (int i = 0; i < n_bases; ++i)
			{
				const auto &bi = vals.basis_values[i];
				for (int m = 0; m < size(); ++m)
				{
					phi_i.setZero();
					phi_i(m) = bi.val(p);

					for (int j = 0; j < n_bases; ++j)
					{
						const auto &bj = vals.basis_values[j];
						for (int n = 0; n < size(); ++n)
						{
							phi_j.setZero();
							phi_j(n) = bj.val(p);
							W(i * size() + m, j * size() + n) += (grad_v * phi_i).dot(phi_j) * da(p);
						}
					}
				}
			}
		}

		return W;
	}

	template class NavierStokesVelocity<true>;
	template class NavierStokesVelocity<false>;
}
