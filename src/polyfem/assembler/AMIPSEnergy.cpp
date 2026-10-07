#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>

#include <polyfem/autogen/elastic_energies/AMIPS2d.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS2drest.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS3d.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS3drest.hpp>

namespace polyfem::assembler
{
	AMIPSEnergy::AMIPSEnergy()
		: use_rest_pose_("use_rest_pose"), weight_("weight")
	{
		autodiff_type_ = AutodiffType::NONE;
	}

	void AMIPSEnergy::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		use_rest_pose_.add_multimaterial(index, params, "", root_path);

		weight_.add_multimaterial(index, params, "", root_path);
	}

	std::map<std::string, Assembler::ParamFunc> AMIPSEnergy::parameters() const
	{
		std::map<std::string, ParamFunc> res;

		res["use_rest_pose"] = [this](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return use_rest_pose_(p, t, e);
		};
		res["weight"] = [this](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return weight_(p, t, e);
		};

		return res;
	}

	bool AMIPSEnergy::use_rest_pose(const RowVectorNd &p, const double t, const int el_id) const
	{
		return use_rest_pose_(p, t, el_id) != 0;
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

		const double weight = weight_(p, t, el_id);

		if (use_rest_pose(p, t, el_id))
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

		const double weight = weight_(p, t, el_id);

		if (use_rest_pose(p, t, el_id))
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