#include "ViscousDamping.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/MatrixUtils.hpp>
#include <igl/Timer.h>

namespace polyfem
{
	namespace assembler
	{
		void ViscousDamping::compute_stress_aux(const Eigen::MatrixXd& F, const Eigen::MatrixXd& dFdt, Eigen::MatrixXd& dRdF, Eigen::MatrixXd& dRdFdot) const
		{
			const int size = F.rows();

			Eigen::MatrixXd dEdt = 0.5 * (dFdt.transpose() * F + F.transpose() * dFdt);
			
			Eigen::MatrixXd tmp = 2 * psi_ * dEdt + phi_ * dEdt.trace() * Eigen::MatrixXd::Identity(size, size);
			dRdF = dFdt * tmp;
			dRdFdot = F * tmp; // Fdot is dFdt
		}

		void ViscousDamping::compute_stress_grad_aux(const Eigen::MatrixXd& F, const Eigen::MatrixXd& dFdt, Eigen::MatrixXd& d2RdF2, Eigen::MatrixXd& d2RdFdFdot, Eigen::MatrixXd& d2RdFdot2) const
		{
			const int size = F.rows();

			Eigen::MatrixXd dEdt = 0.5 * (dFdt.transpose() * F + F.transpose() * dFdt);

			Eigen::MatrixXd dEdotdF, dEdotdFdot;
			dEdotdF.setZero(size * size, size * size);
			dEdotdFdot.setZero(size * size, size * size);
			for (size_t i = 0; i < size; ++i)
			{
				for (size_t j = 0; j < size; ++j)
				{
					for (size_t p = 0; p < size; ++p)
					{
						dEdotdF(i * size + j, p * size + j) += dFdt(p, i) / 2.;
						dEdotdFdot(i * size + j, p * size + j) += F(p, i) / 2.;
						dEdotdF(i * size + j, p * size + i) += dFdt(p, j) / 2.;
						dEdotdFdot(i * size + j, p * size + i) += F(p, j) / 2.;
					}
				}
			}

			d2RdF2.setZero(size * size, size * size);
			d2RdFdFdot.setZero(size * size, size * size);
			d2RdFdot2.setZero(size * size, size * size);
			for (size_t i = 0; i < size; ++i)
			{
				for (size_t j = 0; j < size; ++j)
				{
					const int idx = i * size + j;
					for (size_t k = 0; k < size; ++k)
					{
						d2RdF2.row(idx) += (2 * psi_) * dFdt(i, k) * dEdotdF.row(k * size + j) + phi_ * dFdt(i, j) * dEdotdF.row(k * size + k);
						
						d2RdFdot2.row(idx) += (2 * psi_) * F(i, k) * dEdotdFdot.row(k * size + j) + phi_ * F(i, j) * dEdotdFdot.row(k * size + k);

						d2RdFdFdot(idx, i * size + k) += 2 * psi_ * dEdt(k, j);
						d2RdFdFdot.row(idx) += (2 * psi_) * (dFdt(i, k) * dEdotdFdot.row(k * size + j)) + phi_ * (dEdotdFdot.row(k * size + k) * dFdt(i, j));
						d2RdFdFdot(idx, idx) += phi_ * dEdt(k, k);
					}
				}
			}
		}

		// void ViscousDamping::compute_stress_grad_aux(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Fdot, Eigen::MatrixXd& d2RdF2, Eigen::MatrixXd& d2RdFdFprev) const
		// {
		// 	const int size = F.rows();

		// 	Eigen::MatrixXd dEdt = 0.5 * (Fdot.transpose() * F + F.transpose() * Fdot);

		// 	d2RdF2.setZero(size * size, size * size);
		// 	d2RdFdFprev.setZero(size * size, size * size);
		// }

		ViscousDamping::ViscousDamping()
		{
			set_size(size_);
		}

		void ViscousDamping::set_parameters(const json &params)
		{
			set_size(params["size"]);

			if (params.contains("psi"))
				psi_ = params["psi"];
			if (params.contains("phi"))
				phi_ = params["phi"];
		}

		void ViscousDamping::set_size(const int size)
		{
			size_ = size;
		}

		// E := 0.5(F^T F - I), Compute \int F * (2\psi dE/dt + \phi Tr(dE/dt) I) : gradv du
		Eigen::VectorXd
		ViscousDamping::assemble_prev_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const
		{
			Eigen::MatrixXd local_disp;
			local_disp.setZero(vals.basis_values.size(), size());
			Eigen::MatrixXd local_prev_disp = local_disp;
			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				for (size_t ii = 0; ii < bs.global.size(); ++ii)
				{
					for (int d = 0; d < size(); ++d)
					{
						local_disp(i, d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
						local_prev_disp(i, d) += bs.global[ii].val * prev_displacement(bs.global[ii].index * size() + d);
					}
				}
			}

			Eigen::MatrixXd G;
			G.setZero(vals.basis_values.size(), size());

			const int n_pts = da.size();
			
			Eigen::MatrixXd def_grad(size(), size()), prev_def_grad(size(), size());
			for (long p = 0; p < n_pts; ++p)
			{
				def_grad.setZero();
				prev_def_grad.setZero();

				Eigen::MatrixXd grad(vals.basis_values.size(), size());

				for (size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					grad.row(i) = vals.basis_values[i].grad.row(p);
				}

				Eigen::MatrixXd jac_it = vals.jac_it[p];
				
				def_grad = local_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());
				prev_def_grad = local_prev_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());

				Eigen::MatrixXd delF_delU = grad * jac_it;

				Eigen::MatrixXd dRdF, dRdFdot;
				compute_stress_aux(def_grad, (def_grad - prev_def_grad) / dt, dRdF, dRdFdot);

				G += delF_delU * (dRdFdot / (-dt)) * da(p);
			}

			Eigen::MatrixXd G_T = G.transpose();

			Eigen::VectorXd temp(Eigen::Map<Eigen::VectorXd>(G_T.data(), G_T.size()));

			return temp;
		}

		// E := 0.5(F^T F - I), Compute Stress = \int dF/dt * (2\psi dE/dt + \phi Tr(dE/dt) I) : gradv du
		Eigen::VectorXd
		ViscousDamping::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const
		{
			Eigen::MatrixXd local_disp;
			local_disp.setZero(vals.basis_values.size(), size());
			Eigen::MatrixXd local_prev_disp = local_disp;
			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				for (size_t ii = 0; ii < bs.global.size(); ++ii)
				{
					for (int d = 0; d < size(); ++d)
					{
						local_disp(i, d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
						local_prev_disp(i, d) += bs.global[ii].val * prev_displacement(bs.global[ii].index * size() + d);
					}
				}
			}

			Eigen::MatrixXd G;
			G.setZero(vals.basis_values.size(), size());
			
			const int n_pts = da.size();

			Eigen::MatrixXd def_grad(size(), size()), prev_def_grad(size(), size());
			for (long p = 0; p < n_pts; ++p)
			{
				def_grad.setZero();
				prev_def_grad.setZero();

				Eigen::MatrixXd grad(vals.basis_values.size(), size());

				for (size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					grad.row(i) = vals.basis_values[i].grad.row(p);
				}

				Eigen::MatrixXd jac_it = vals.jac_it[p];
				
				def_grad = local_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());
				prev_def_grad = local_prev_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());

				Eigen::MatrixXd delF_delU = grad * jac_it;
				auto dFdt = (def_grad - prev_def_grad) / dt;

				Eigen::MatrixXd dRdF, dRdFdot;
				Eigen::MatrixXd dEdt = 0.5 * (dFdt.transpose() * def_grad + def_grad.transpose() * dFdt);
				Eigen::MatrixXd tmp = 2 * psi_ * dEdt + phi_ * dEdt.trace() * Eigen::MatrixXd::Identity(size(), size());

				G += delF_delU * ((dFdt + def_grad / dt) * tmp).transpose() * da(p);
			}

			Eigen::MatrixXd G_T = G.transpose();

			Eigen::VectorXd temp(Eigen::Map<Eigen::VectorXd>(G_T.data(), G_T.size()));

			return temp;
		}

		// Compute \int grad phi_i : dStress / dF^n : grad phi_j dx
		Eigen::MatrixXd
		ViscousDamping::assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const
		{
			Eigen::MatrixXd hessian;
			hessian.setZero(vals.basis_values.size() * size(), vals.basis_values.size() * size());

			Eigen::MatrixXd local_disp;
			local_disp.setZero(vals.basis_values.size(), size());
			Eigen::MatrixXd local_prev_disp = local_disp;
			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				for (size_t ii = 0; ii < bs.global.size(); ++ii)
				{
					for (int d = 0; d < size(); ++d)
					{
						local_disp(i, d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
						local_prev_disp(i, d) += bs.global[ii].val * prev_displacement(bs.global[ii].index * size() + d);
					}
				}
			}

			const int n_pts = da.size();

			Eigen::MatrixXd def_grad(size(), size()), prev_def_grad(size(), size());
			Eigen::MatrixXd d2RdF2, d2RdFdFdot, d2RdFdot2;
			Eigen::MatrixXd hessian_temp, hessian_temp2;
			for (long p = 0; p < n_pts; ++p)
			{
				Eigen::MatrixXd grad(vals.basis_values.size(), size());

				for (size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					grad.row(i) = vals.basis_values[i].grad.row(p);
				}

				Eigen::MatrixXd jac_it = vals.jac_it[p];
				
				def_grad = local_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());
				prev_def_grad = local_prev_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());

				Eigen::MatrixXd dFdt = (def_grad - prev_def_grad) / dt;
				compute_stress_grad_aux(def_grad, dFdt, d2RdF2, d2RdFdFdot, d2RdFdot2);

				hessian_temp = d2RdF2 + (1. / dt) * (d2RdFdFdot + d2RdFdFdot.transpose()) + (1. / dt / dt) * d2RdFdot2;
				hessian_temp2 = hessian_temp;
				for (int i = 0; i < size(); i++)
				for (int j = 0; j < size(); j++)
				for (int k = 0; k < size(); k++)
				for (int l = 0; l < size(); l++)
					hessian_temp(i + j * size(), k + l * size()) = hessian_temp2(i * size() + j, k * size() + l);

				Eigen::MatrixXd delF_delU_tensor(jac_it.size(), grad.size());

				for (size_t i = 0; i < local_disp.rows(); ++i)
				{
					for (size_t j = 0; j < local_disp.cols(); ++j)
					{
						Eigen::MatrixXd temp;
						temp.setZero(size(), size());
						temp.row(j) = grad.row(i);
						temp = temp * jac_it;
						Eigen::VectorXd temp_flattened(Eigen::Map<Eigen::VectorXd>(temp.data(), temp.size()));
						delF_delU_tensor.col(i * size() + j) = temp_flattened;
					}
				}

				hessian += delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor * da(p);
			}

			return hessian;
		}

		// Compute \int grad phi_i : dStress / dF^{n-1} : grad phi_j dx
		Eigen::MatrixXd
		ViscousDamping::assemble_stress_prev_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const
		{
			Eigen::MatrixXd stress_grad_Ut;
			stress_grad_Ut.setZero(vals.basis_values.size() * size(), vals.basis_values.size() * size());

			Eigen::MatrixXd local_disp;
			local_disp.setZero(vals.basis_values.size(), size());
			Eigen::MatrixXd local_prev_disp = local_disp;
			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				for (size_t ii = 0; ii < bs.global.size(); ++ii)
				{
					for (int d = 0; d < size(); ++d)
					{
						local_disp(i, d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
						local_prev_disp(i, d) += bs.global[ii].val * prev_displacement(bs.global[ii].index * size() + d);
					}
				}
			}

			const int n_pts = da.size();

			Eigen::MatrixXd def_grad(size(), size()), prev_def_grad(size(), size());
			for (long p = 0; p < n_pts; ++p)
			{
				def_grad.setZero();
				prev_def_grad.setZero();

				Eigen::MatrixXd grad(vals.basis_values.size(), size());

				for (size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					grad.row(i) = vals.basis_values[i].grad.row(p);
				}

				Eigen::MatrixXd jac_it = vals.jac_it[p];
				
				def_grad = local_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());
				prev_def_grad = local_prev_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());

				Eigen::MatrixXd d2RdF2, d2RdFdFdot, d2RdFdot2;
				Eigen::MatrixXd dFdt = (def_grad - prev_def_grad) / dt;
				compute_stress_grad_aux(def_grad, dFdt, d2RdF2, d2RdFdFdot, d2RdFdot2);

				Eigen::MatrixXd stress_grad_Ut_temp = - (1. / dt) * d2RdFdFdot - (1. / dt / dt) * d2RdFdot2;
				Eigen::MatrixXd stress_grad_Ut_temp2 = stress_grad_Ut_temp;
				for (int i = 0; i < size(); i++)
				for (int j = 0; j < size(); j++)
				for (int k = 0; k < size(); k++)
				for (int l = 0; l < size(); l++)
					stress_grad_Ut_temp(i + j * size(), k + l * size()) = stress_grad_Ut_temp2(i * size() + j, k * size() + l);

				Eigen::MatrixXd delF_delU_tensor(jac_it.size(), grad.size());

				for (size_t i = 0; i < local_disp.rows(); ++i)
				{
					for (size_t j = 0; j < local_disp.cols(); ++j)
					{
						Eigen::MatrixXd temp;
						temp.setZero(size(), size());
						temp.row(j) = grad.row(i);
						temp = temp * jac_it;
						Eigen::VectorXd temp_flattened(Eigen::Map<Eigen::VectorXd>(temp.data(), temp.size()));
						delF_delU_tensor.col(i * size() + j) = temp_flattened;
					}
				}

				stress_grad_Ut += delF_delU_tensor.transpose() * stress_grad_Ut_temp * delF_delU_tensor * da(p);
			}

			return stress_grad_Ut;
		}

		// E := 0.5(F^T F - I), Compute Energy = \int \psi \| \frac{\partial E}{\partial t} \|^2 + 0.5 \phi (Tr(\frac{\partial E}{\partial t}))^2 du
		double ViscousDamping::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::MatrixXd &prev_displacement, const QuadratureVector &da, const double dt) const
		{
			Eigen::MatrixXd local_disp;
			local_disp.setZero(vals.basis_values.size(), size());
			Eigen::MatrixXd local_prev_disp = local_disp;
			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				for (size_t ii = 0; ii < bs.global.size(); ++ii)
				{
					for (int d = 0; d < size(); ++d)
					{
						local_disp(i, d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
						local_prev_disp(i, d) += bs.global[ii].val * prev_displacement(bs.global[ii].index * size() + d);
					}
				}
			}
			
			double energy = 0;
			const int n_pts = da.size();

			Eigen::MatrixXd def_grad(size(), size()), prev_def_grad(size(), size());
			for (long p = 0; p < n_pts; ++p)
			{
				def_grad.setZero();
				prev_def_grad.setZero();

				for (size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					const auto &bs = vals.basis_values[i];

					for (int d = 0; d < size(); ++d)
					{
						for (int c = 0; c < size(); ++c)
						{
							def_grad(d, c) += bs.grad(p, c) * local_disp(i, d);
							prev_def_grad(d, c) += bs.grad(p, c) * local_prev_disp(i, d);
						}
					}
				}

				Eigen::MatrixXd jac_it(size(), size());
				for (long k = 0; k < jac_it.size(); ++k)
					jac_it(k) = vals.jac_it[p](k);
				
				def_grad = def_grad * jac_it + Eigen::MatrixXd::Identity(size(), size());
				prev_def_grad = prev_def_grad * jac_it + Eigen::MatrixXd::Identity(size(), size());

				Eigen::MatrixXd dFdt = (def_grad - prev_def_grad) / dt;
				Eigen::MatrixXd dEdt = 0.5 * (dFdt.transpose() * def_grad + def_grad.transpose() * dFdt);
				
				double val = psi_ * dEdt.squaredNorm() + 0.5 * phi_ * pow(dEdt.trace(), 2);

				energy += val * da(p);
			}

			return energy;
		}

		void ViscousDamping::compute_stress_grad(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const
		{
			Eigen::MatrixXd F = Eigen::MatrixXd::Identity(size(), size()) + grad_u_i;
			Eigen::MatrixXd Fdot = (grad_u_i - prev_grad_u_i) / dt;
			Eigen::MatrixXd dRdF, dRdFdot, d2RdF2, d2RdFdFdot, d2RdFdot2;
			
			// compute_stress_aux(F, Fdot, dRdF, dRdFdot);
			Eigen::MatrixXd dEdt = 0.5 * (Fdot.transpose() * F + F.transpose() * Fdot);
			Eigen::MatrixXd tmp = 2 * psi_ * dEdt + phi_ * dEdt.trace() * Eigen::MatrixXd::Identity(size(), size());
			stress = (Fdot + (1. / dt) * F) * tmp;

			compute_stress_grad_aux(F, Fdot, d2RdF2, d2RdFdFdot, d2RdFdot2);
			result = d2RdF2 + (1. / dt) * (d2RdFdFdot + d2RdFdFdot.transpose()) + (1. / dt / dt) * d2RdFdot2;
		}

		void ViscousDamping::compute_stress_prev_grad(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &result) const
		{
			Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(size(), size()) + grad_u_i;
			Eigen::MatrixXd def_grad_fd = (grad_u_i - prev_grad_u_i) / dt;
			Eigen::MatrixXd d2RdF2, d2RdFdFdot, d2RdFdot2;

			compute_stress_grad_aux(def_grad, def_grad_fd, d2RdF2, d2RdFdFdot, d2RdFdot2);
			result = - (1. / dt) * d2RdFdFdot - (1. / dt / dt) * d2RdFdot2;
		}

		void ViscousDamping::compute_dstress_dpsi_dphi(const int el_id, const double dt, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &prev_grad_u_i, Eigen::MatrixXd &dstress_dpsi, Eigen::MatrixXd &dstress_dphi) const
		{
			Eigen::MatrixXd F = Eigen::MatrixXd::Identity(size(), size()) + grad_u_i;
			Eigen::MatrixXd prev_F = Eigen::MatrixXd::Identity(size(), size()) + prev_grad_u_i;

			Eigen::MatrixXd dFdt = (F - prev_F) / dt;
			Eigen::MatrixXd dEdt = 0.5 * (dFdt.transpose() * F + F.transpose() * dFdt);

			dstress_dpsi = 2 * (dFdt + F / dt) * dEdt;
			dstress_dphi = dEdt.trace() * (dFdt + F / dt);
		}
	}
} // namespace polyfem
