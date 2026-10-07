#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>

#include <polyfem/autogen/elastic_energies/AMIPS2d.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS2drest.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS3d.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS3drest.hpp>

namespace polyfem::assembler
{
	void AMIPSEnergy::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		if (use_rest_pose_.size() <= index)
			use_rest_pose_.resize(index + 1, false);

		use_rest_pose_[index] = params.contains("use_rest_pose") && params["use_rest_pose"].get<bool>();

		if (energy_weights_.size() <= index)
			energy_weights_.resize(index + 1, 1.0);

		if (params.contains("weight"))
			energy_weights_[index] = params["weight"].get<double>();
		else
			energy_weights_[index] = 1.0;

		if (powers_.size() <= index)
			powers_.resize(index + 1, 1.0);

		if (params.contains("power"))
			powers_[index] = params["power"].get<double>();
		else
			powers_[index] = 1.0;
		// A power <= 0 defeats the energy: negative rewards distortion, and at 0 pow(NaN, 0) = 1
		// gives an inverted element a finite value, so the line search no longer refuses it.
		if (powers_[index] <= 0)
			log_and_throw_error("AMIPS power must be > 0, got {}", powers_[index]);
	}

	double AMIPSEnergy::get_energy_weight(const int el_id) const
	{
		if (energy_weights_.empty())
			return 1.0;
		if (energy_weights_.size() == 1)
			return energy_weights_[0];
		if (el_id >= 0 && el_id < energy_weights_.size())
			return energy_weights_[el_id];
		return 1.0;
	}

	bool AMIPSEnergy::use_rest_pose(const int el_id) const
	{
		if (use_rest_pose_.empty())
			return false;
		if (use_rest_pose_.size() == 1)
			return use_rest_pose_[0];
		if (el_id >= 0 && el_id < (int)use_rest_pose_.size())
			return use_rest_pose_[el_id];
		return false;
	}

	double AMIPSEnergy::get_power(const int el_id) const
	{
		if (powers_.empty())
			return 1.0;
		if (powers_.size() == 1)
			return powers_[0];
		if (el_id >= 0 && el_id < powers_.size())
			return powers_[el_id];
		return 1.0;
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPSEnergy::amips_gradient(
		const RowVectorNd &p,
		const double t,
		const int el_id,
		const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
	{
		if (use_rest_pose(el_id))
		{
			if (size() == 2)
				return autogen::AMIPS2drest_gradient(p, t, el_id, F);
			else
				return autogen::AMIPS3drest_gradient(p, t, el_id, F);
		}
		else
		{
			if (size() == 2)
				return autogen::AMIPS2d_gradient(p, t, el_id, F);
			else
				return autogen::AMIPS3d_gradient(p, t, el_id, F);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPSEnergy::gradient(
		const RowVectorNd &p,
		const double t,
		const int el_id,
		const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
	{
		const double det = polyfem::utils::determinant(F);
		if (det <= 0)
		{
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(size(), size());
			grad.setConstant(std::nan(""));
			return grad;
		}

		const double weight = get_energy_weight(el_id);
		const double power = get_power(el_id);

		if (power == 1)
			return weight * amips_gradient(p, t, el_id, F);

		// grad(psi^power) = power psi^(power - 1) grad psi
		DefGradMatrix<double> def_grad = F;
		const double psi = amips(el_id, 1., def_grad);
		return weight * power * std::pow(psi, power - 1) * amips_gradient(p, t, el_id, F);
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPSEnergy::hessian(
		const RowVectorNd &p,
		const double t,
		const int el_id,
		const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
	{
		const double det = polyfem::utils::determinant(F);
		if (det <= 0)
		{
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hessian(size() * size(), size() * size());
			hessian.setConstant(std::nan(""));
			return hessian;
		}

		const double weight = get_energy_weight(el_id);

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hess;
		if (use_rest_pose(el_id))
		{
			if (size() == 2)
				hess = autogen::AMIPS2drest_hessian(p, t, el_id, F);
			else
				hess = autogen::AMIPS3drest_hessian(p, t, el_id, F);
		}
		else
		{
			if (size() == 2)
				hess = autogen::AMIPS2d_hessian(p, t, el_id, F);
			else
				hess = autogen::AMIPS3d_hessian(p, t, el_id, F);
		}

		const double power = get_power(el_id);
		if (power != 1)
		{
			// hess(psi^power) = power psi^(power - 1) hess psi + power (power - 1) psi^(power - 2) grad psi grad psi^T,
			// grad psi flattened row-major like the hessian's index i * size() + j.
			DefGradMatrix<double> def_grad = F;
			const double psi = amips(el_id, 1., def_grad);
			const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> g = amips_gradient(p, t, el_id, F).reshaped<Eigen::RowMajor>();
			hess = power * std::pow(psi, power - 1) * hess + power * (power - 1) * std::pow(psi, power - 2) * g * g.transpose();
		}

		return weight * hess;
	}

} // namespace polyfem::assembler