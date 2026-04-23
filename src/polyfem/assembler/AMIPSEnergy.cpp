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

		if (params.contains("use_rest_pose"))
		{
			use_rest_pose_ = params["use_rest_pose"].get<bool>();
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

		if (use_rest_pose_)
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

		if (use_rest_pose_)
		{
			if (size() == 2)
				return autogen::AMIPS2drest_hessian(p, t, el_id, F);
			else
				return autogen::AMIPS3drest_hessian(p, t, el_id, F);
		}
		else
		{
			if (size() == 2)
				return autogen::AMIPS2d_hessian(p, t, el_id, F);
			else
				return autogen::AMIPS3d_hessian(p, t, el_id, F);
		}
	}

} // namespace polyfem::assembler