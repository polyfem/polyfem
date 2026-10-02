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
	}

	double AMIPSEnergy::get_energy_weight(const int el_id) const
	{
		if (energy_weights_.empty())
			return 1.0;
		if (energy_weights_.size() == 1)
			return energy_weights_[0];
		if (el_id >= 0 && el_id < (int)energy_weights_.size())
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

		if (use_rest_pose(el_id))
		{
			if (size() == 2)
				return weight * autogen::AMIPS2drest_gradient(p, t, el_id, F);
			else
				return weight * autogen::AMIPS3drest_gradient(p, t, el_id, F);
		}
		else
		{
			if (size() == 2)
				return weight * autogen::AMIPS2d_gradient(p, t, el_id, F);
			else
				return weight * autogen::AMIPS3d_gradient(p, t, el_id, F);
		}
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

		if (use_rest_pose(el_id))
		{
			if (size() == 2)
				return weight * autogen::AMIPS2drest_hessian(p, t, el_id, F);
			else
				return weight * autogen::AMIPS3drest_hessian(p, t, el_id, F);
		}
		else
		{
			if (size() == 2)
				return weight * autogen::AMIPS2d_hessian(p, t, el_id, F);
			else
				return weight * autogen::AMIPS3d_hessian(p, t, el_id, F);
		}
	}

} // namespace polyfem::assembler