#include "ViscousDamping.hpp"

namespace polyfem::assembler
{
	void ViscousDamping::compute_stress_aux(const Eigen::MatrixXd &F, const Eigen::MatrixXd &dFdt, Eigen::MatrixXd &dRdF, Eigen::MatrixXd &dRdFdot) const
	{
		const int size = F.rows();

		Eigen::MatrixXd dEdt = 0.5 * (dFdt.transpose() * F + F.transpose() * dFdt);

		Eigen::MatrixXd tmp = 2 * damping_params_[0] * dEdt + damping_params_[1] * dEdt.trace() * Eigen::MatrixXd::Identity(size, size);
		dRdF = dFdt * tmp;
		dRdFdot = F * tmp; // Fdot is dFdt
	}

	void ViscousDamping::compute_stress_grad_aux(const Eigen::MatrixXd &F, const Eigen::MatrixXd &dFdt, Eigen::MatrixXd &d2RdF2, Eigen::MatrixXd &d2RdFdFdot, Eigen::MatrixXd &d2RdFdot2) const
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
					d2RdF2.row(idx) += (2 * damping_params_[0]) * dFdt(i, k) * dEdotdF.row(k * size + j) + damping_params_[1] * dFdt(i, j) * dEdotdF.row(k * size + k);

					d2RdFdot2.row(idx) += (2 * damping_params_[0]) * F(i, k) * dEdotdFdot.row(k * size + j) + damping_params_[1] * F(i, j) * dEdotdFdot.row(k * size + k);

					d2RdFdFdot(idx, i * size + k) += 2 * damping_params_[0] * dEdt(k, j);
					d2RdFdFdot.row(idx) += (2 * damping_params_[0]) * (dFdt(i, k) * dEdotdFdot.row(k * size + j)) + damping_params_[1] * (dEdotdFdot.row(k * size + k) * dFdt(i, j));
					d2RdFdFdot(idx, idx) += damping_params_[1] * dEdt(k, k);
				}
			}
		}
	}

	void ViscousDamping::add_multimaterial(const int index, const json &params, const Units &units)
	{
		// TODO add units
		assert(size() == 2 || size() == 3);

		if (params.contains("psi"))
			damping_params_[0] = params["psi"];
		if (params.contains("phi"))
			damping_params_[1] = params["phi"];
	}

	// E := 0.5(F^T F - I), Compute \int F * (2\psi dE/dt + \phi Tr(dE/dt) I) : gradv du
	Eigen::VectorXd
	ViscousDampingPrev::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		if (data.x_prev.size() != data.x.size())
			return Eigen::VectorXd::Zero(data.vals.basis_values.size() * size());

		Eigen::MatrixXd local_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		Eigen::MatrixXd local_prev_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				local_disp.row(i) += bs.global[ii].val * data.x.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
				local_prev_disp.row(i) += bs.global[ii].val * data.x_prev.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
			}
		}

		Eigen::MatrixXd local_vel = (local_disp - local_prev_disp) / data.dt;

		Eigen::MatrixXd G;
		G.setZero(data.vals.basis_values.size(), size());

		const int n_pts = data.da.size();

		Eigen::MatrixXd def_grad, vel_def_grad;
		Eigen::MatrixXd delF_delU;
		Eigen::MatrixXd dRdF, dRdFdot;
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			delF_delU = grad * data.vals.jac_it[p];
			def_grad = local_disp.transpose() * delF_delU + Eigen::MatrixXd::Identity(size(), size());
			vel_def_grad = local_vel.transpose() * delF_delU;

			compute_stress_aux(def_grad, vel_def_grad, dRdF, dRdFdot);
			G -= delF_delU * dRdFdot * (data.da(p) / data.dt);
		}

		G.transposeInPlace();

		Eigen::VectorXd temp(Eigen::Map<Eigen::VectorXd>(G.data(), G.size()));

		return temp;
	}

	// E := 0.5(F^T F - I), Compute Stress = \int dF/dt * (2\psi dE/dt + \phi Tr(dE/dt) I) : gradv du
	Eigen::VectorXd
	ViscousDamping::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		if (data.x_prev.size() != data.x.size())
			return Eigen::VectorXd::Zero(data.vals.basis_values.size() * size());

		Eigen::MatrixXd local_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		Eigen::MatrixXd local_prev_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				local_disp.row(i) += bs.global[ii].val * data.x.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
				local_prev_disp.row(i) += bs.global[ii].val * data.x_prev.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
			}
		}

		Eigen::MatrixXd local_vel = (local_disp - local_prev_disp) / data.dt;

		Eigen::MatrixXd G;
		G.setZero(data.vals.basis_values.size(), size());

		const int n_pts = data.da.size();

		Eigen::MatrixXd def_grad, dFdt, dEdt;
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::MatrixXd delF_delU = grad * data.vals.jac_it[p];

			def_grad = local_disp.transpose() * delF_delU + Eigen::MatrixXd::Identity(size(), size());
			dFdt = local_vel.transpose() * delF_delU;

			dEdt = dFdt.transpose() * def_grad;
			dEdt = (dEdt + dEdt.transpose()).eval() / 2.;

			Eigen::MatrixXd tmp = (2 * damping_params_[0]) * dEdt + (damping_params_[1] * dEdt.trace()) * Eigen::MatrixXd::Identity(size(), size());
			G += delF_delU * ((dFdt + def_grad / data.dt) * tmp).transpose() * data.da(p);
		}

		G.transposeInPlace();

		Eigen::VectorXd temp(Eigen::Map<Eigen::VectorXd>(G.data(), G.size()));

		return temp;
	}

	// Compute \int grad phi_i : dStress / dF^n : grad phi_j dx
	Eigen::MatrixXd
	ViscousDamping::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		Eigen::MatrixXd hessian;
		hessian.setZero(data.vals.basis_values.size() * size(), data.vals.basis_values.size() * size());
		if (data.x_prev.size() != data.x.size())
			return hessian;

		Eigen::MatrixXd local_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		Eigen::MatrixXd local_prev_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				local_disp.row(i) += bs.global[ii].val * data.x.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
				local_prev_disp.row(i) += bs.global[ii].val * data.x_prev.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
			}
		}

		Eigen::MatrixXd local_vel = (local_disp - local_prev_disp) / data.dt;

		const int n_pts = data.da.size();

		Eigen::MatrixXd def_grad, dFdt;
		Eigen::MatrixXd d2RdF2, d2RdFdFdot, d2RdFdot2;
		Eigen::MatrixXd hessian_temp, delF_delU_tensor;
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::MatrixXd delF_delU = grad * data.vals.jac_it[p];

			def_grad = local_disp.transpose() * delF_delU + Eigen::MatrixXd::Identity(size(), size());
			dFdt = local_vel.transpose() * delF_delU;

			compute_stress_grad_aux(def_grad, dFdt, d2RdF2, d2RdFdFdot, d2RdFdot2);
			hessian_temp = d2RdF2 + (1. / data.dt) * (d2RdFdFdot + d2RdFdFdot.transpose()) + (1. / data.dt / data.dt) * d2RdFdot2;

			delF_delU_tensor = Eigen::MatrixXd::Zero(size() * size(), grad.size());
			for (size_t i = 0; i < local_disp.rows(); ++i)
				for (size_t j = 0; j < size(); ++j)
					for (size_t d = 0; d < size(); d++)
						delF_delU_tensor(size() * j + d, i * size() + j) = delF_delU(i, d);

			hessian += delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor * data.da(p);
		}

		return hessian;
	}

	// Compute \int grad phi_i : dStress / dF^{n-1} : grad phi_j dx
	Eigen::MatrixXd
	ViscousDampingPrev::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		Eigen::MatrixXd stress_grad_Ut;
		stress_grad_Ut.setZero(data.vals.basis_values.size() * size(), data.vals.basis_values.size() * size());
		if (data.x_prev.size() != data.x.size())
			return stress_grad_Ut;

		Eigen::MatrixXd local_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		Eigen::MatrixXd local_prev_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				local_disp.row(i) += bs.global[ii].val * data.x.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
				local_prev_disp.row(i) += bs.global[ii].val * data.x_prev.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
			}
		}

		Eigen::MatrixXd local_vel = (local_disp - local_prev_disp) / data.dt;

		const int n_pts = data.da.size();

		Eigen::MatrixXd def_grad, dFdt;
		Eigen::MatrixXd d2RdF2, d2RdFdFdot, d2RdFdot2;
		Eigen::MatrixXd stress_grad_Ut_temp, delF_delU_tensor;
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			Eigen::MatrixXd delF_delU = grad * data.vals.jac_it[p];

			def_grad = local_disp.transpose() * delF_delU + Eigen::MatrixXd::Identity(size(), size());
			dFdt = local_vel.transpose() * delF_delU;

			compute_stress_grad_aux(def_grad, dFdt, d2RdF2, d2RdFdFdot, d2RdFdot2);
			stress_grad_Ut_temp = -(1. / data.dt) * d2RdFdFdot - (1. / data.dt / data.dt) * d2RdFdot2;

			delF_delU_tensor = Eigen::MatrixXd::Zero(size() * size(), grad.size());
			for (size_t i = 0; i < local_disp.rows(); ++i)
				for (size_t j = 0; j < size(); ++j)
					for (size_t d = 0; d < size(); d++)
						delF_delU_tensor(size() * j + d, i * size() + j) = delF_delU(i, d);

			stress_grad_Ut += delF_delU_tensor.transpose() * stress_grad_Ut_temp * delF_delU_tensor * data.da(p);
		}

		return stress_grad_Ut;
	}

	// E := 0.5(F^T F - I), Compute Energy = \int \psi \| \frac{\partial E}{\partial t} \|^2 + 0.5 \phi (Tr(\frac{\partial E}{\partial t}))^2 du
	double ViscousDamping::compute_energy(const NonLinearAssemblerData &data) const
	{
		if (data.x_prev.size() != data.x.size())
			return 0;

		Eigen::MatrixXd local_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		Eigen::MatrixXd local_prev_disp = Eigen::MatrixXd::Zero(data.vals.basis_values.size(), size());
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				local_disp.row(i) += bs.global[ii].val * data.x.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
				local_prev_disp.row(i) += bs.global[ii].val * data.x_prev.block(bs.global[ii].index * size(), 0, size(), 1).transpose();
			}
		}

		Eigen::MatrixXd local_vel = (local_disp - local_prev_disp) / data.dt;

		double energy = 0;
		const int n_pts = data.da.size();

		Eigen::MatrixXd def_grad, dFdt, dEdt;
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(data.vals.basis_values.size(), size());

			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				grad.row(i) = data.vals.basis_values[i].grad.row(p);
			}

			const Eigen::MatrixXd delF_delU = grad * data.vals.jac_it[p];
			def_grad = local_disp.transpose() * delF_delU + Eigen::MatrixXd::Identity(size(), size());
			dFdt = local_vel.transpose() * delF_delU;

			dEdt = dFdt.transpose() * def_grad;
			dEdt = (dEdt + dEdt.transpose()).eval() / 2.;

			double val = damping_params_[0] * dEdt.squaredNorm() + 0.5 * damping_params_[1] * pow(dEdt.trace(), 2);
			energy += val * data.da(p);
		}

		return energy;
	}

	void ViscousDamping::compute_stress_grad(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &prev_grad_u_i,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		const double dt = data.dt;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		const Eigen::MatrixXd F = Eigen::MatrixXd::Identity(size(), size()) + grad_u_i;
		const Eigen::MatrixXd Fdot = (grad_u_i - prev_grad_u_i) / dt;
		Eigen::MatrixXd d2RdF2, d2RdFdFdot, d2RdFdot2;

		Eigen::MatrixXd dEdt = Fdot.transpose() * F;
		dEdt = (dEdt + dEdt.transpose()).eval() / 2.;

		Eigen::MatrixXd tmp = (2 * damping_params_[0]) * dEdt + (damping_params_[1] * dEdt.trace()) * Eigen::MatrixXd::Identity(size(), size());
		stress = (Fdot + (1. / dt) * F) * tmp;

		compute_stress_grad_aux(F, Fdot, d2RdF2, d2RdFdFdot, d2RdFdot2);
		result = d2RdF2 + (1. / dt) * (d2RdFdFdot + d2RdFdFdot.transpose()) + (1. / dt / dt) * d2RdFdot2;
	}

	void ViscousDamping::compute_stress_prev_grad(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &prev_grad_u_i,
		Eigen::MatrixXd &result) const
	{
		const double dt = data.dt;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		const Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
		const Eigen::MatrixXd def_grad_fd = (grad_u_i - prev_grad_u_i) / dt;
		Eigen::MatrixXd d2RdF2, d2RdFdFdot, d2RdFdot2;

		compute_stress_grad_aux(def_grad, def_grad_fd, d2RdF2, d2RdFdFdot, d2RdFdot2);
		result = -(1. / dt) * d2RdFdFdot - (1. / dt / dt) * d2RdFdot2;
	}

	void ViscousDamping::compute_dstress_dpsi_dphi(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &prev_grad_u_i,
		Eigen::MatrixXd &dstress_dpsi,
		Eigen::MatrixXd &dstress_dphi)
	{
		const double dt = data.dt;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		const Eigen::MatrixXd F = Eigen::MatrixXd::Identity(grad_u_i.rows(), grad_u_i.cols()) + grad_u_i;
		const Eigen::MatrixXd dFdt = (grad_u_i - prev_grad_u_i) / dt;
		const Eigen::MatrixXd dEdt = 0.5 * (dFdt.transpose() * F + F.transpose() * dFdt);

		dstress_dpsi = 2 * (dFdt + F / dt) * dEdt;
		dstress_dphi = dEdt.trace() * (dFdt + F / dt);
	}
} // namespace polyfem::assembler
