#include <polyfem/NavierStokes.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElasticityUtils.hpp>

namespace polyfem
{
	void NavierStokesVelocity::set_parameters(const json &params)
	{
		set_size(params["size"]);

		if (params.count("viscosity")) {
			viscosity_ = params["viscosity"];
		}
	}

	void NavierStokesVelocity::set_size(const int size)
	{
		size_ = size;
	}



	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	NavierStokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		//TODO
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());
		assert(false);
		return res;
	}

	void NavierStokesVelocity::compute_norm_velocity(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const
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

	void NavierStokesVelocity::compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const
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

	Eigen::VectorXd
	NavierStokesVelocity::assemble(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
	{
		return compute_grad_aux<double>(vals, velocity, da);
	}

	Eigen::MatrixXd
	NavierStokesVelocity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
	{
		// typedef DScalar1<double, Eigen::VectorXd> GradDScalarT;
		// const auto tmp = compute_grad_aux<GradDScalarT>(vals, velocity, da);
		// Eigen::MatrixXd H(tmp.size(), tmp.size());
		// for(int i = 0; i < tmp.size(); ++i)
		// 	H.col(i) = tmp(i).getGradient();

		// // std::cout<<H<<std::endl;
		// Eigen::MatrixXd Hh = H;



		// Eigen::MatrixXd H = compute_N(vals, velocity, da) + compute_W(vals, velocity, da);

		//For picard
		Eigen::MatrixXd H = compute_N(vals, velocity, da);

		return H.transpose();
	}

	//Compute N = int v \cdot \nabla phi_i \cdot \phi_j
	Eigen::MatrixXd NavierStokesVelocity::compute_N(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
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
	Eigen::MatrixXd NavierStokesVelocity::compute_W(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
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

	//Compute \int (v . grad v, psi)
	template <typename T>
	Eigen::Matrix<T, Eigen::Dynamic, 1> NavierStokesVelocity::compute_grad_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &velocity, const QuadratureVector &da) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

		assert(velocity.cols() == 1);

		const int n_pts = da.size();
		const int n_bases = vals.basis_values.size();


		Eigen::Matrix<double, Eigen::Dynamic, 1> local_velocityv(n_bases * size(), 1);
		local_velocityv.setZero();
		for (size_t i = 0; i < n_bases; ++i)
		{
			const auto &bs = vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_velocityv(i * size() + d) += bs.global[ii].val * velocity(bs.global[ii].index * size() + d);
				}
			}
		}

		const AutoDiffAllocator<T> allocate_auto_diff_scalar;
		AutoDiffVect local_vel(local_velocityv.rows(), 1);
		DiffScalarBase::setVariableCount(local_velocityv.rows());

		for (long i = 0; i < local_velocityv.rows(); ++i)
		{
			local_vel(i) = allocate_auto_diff_scalar(i, local_velocityv(i));
		}


		Eigen::Matrix<T, Eigen::Dynamic, 1> res_grad(size() * n_bases, 1);
		for(int i = 0; i < res_grad.size(); ++i)
			res_grad(i) = T(0.0);

		AutoDiffGradMat grad_v(size(), size());
		AutoDiffVect vel(size(), 1);

		for (long p = 0; p < n_pts; ++p)
		{
			for (long k = 0; k < grad_v.size(); ++k)
				grad_v(k) = T(0);
			for (long k = 0; k < vel.size(); ++k)
				vel(k) = T(0);

			for (size_t i = 0; i < n_bases; ++i)
			{
				const auto &bs = vals.basis_values[i];
				const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad = bs.grad.row(p);
				const double val = bs.val(p);

				assert(grad.size() == size());

				for (int d = 0; d < size(); ++d)
				{
					vel(d) += val * local_vel(i * size() + d);

					for (int c = 0; c < size(); ++c)
					{
						grad_v(d, c) += grad(c) * local_vel(i * size() + d);
					}
				}
			}

			AutoDiffGradMat jac_it(size(), size());
			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = T(vals.jac_it[p](k));
			grad_v = (grad_v * jac_it).eval();

			AutoDiffVect tmp = grad_v.transpose() * vel;

			for (int j = 0; j < n_bases; ++j)
			{
				const auto &bs = vals.basis_values[j];
				const double val = bs.val(p);
				for (int d = 0; d < size(); ++d)
				{
					res_grad(j * size() + d) += tmp(d)*val*da(p);
				}
			}
		}

		return res_grad;
	}
}
