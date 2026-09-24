#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>

#include <polyfem/autogen/elastic_energies/AMIPS2d.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS2drest.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS3d.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS3drest.hpp>

namespace polyfem::assembler
{
	namespace
	{
		template <typename Derived>
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> autodiff_gradient(
			const Derived &self,
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F)
		{
			typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>> Diff;
			typedef Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

			const int size = self.size();
			DiffScalarBase::setVariableCount(size * size);

			AutoDiffGradMat def_grad(size, size);
			for (int i = 0; i < size; ++i)
				for (int j = 0; j < size; ++j)
					def_grad(i, j) = Diff(i * size + j, F(i, j));

			const Diff val = self.template elastic_energy<Diff>(p, t, el_id, def_grad);

			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(size, size);
			for (int i = 0; i < size; ++i)
				for (int j = 0; j < size; ++j)
					grad(i, j) = val.getGradient()(i * size + j);

			return grad;
		}

		template <typename Derived>
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> autodiff_hessian(
			const Derived &self,
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F)
		{
			typedef DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9>> Diff2;
			typedef Eigen::Matrix<Diff2, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

			const int size = self.size();
			DiffScalarBase::setVariableCount(size * size);

			AutoDiffGradMat def_grad(size, size);
			for (int i = 0; i < size; ++i)
				for (int j = 0; j < size; ++j)
					def_grad(i, j) = Diff2(i * size + j, F(i, j));

			const Diff2 val = self.template elastic_energy<Diff2>(p, t, el_id, def_grad);

			return val.getHessian();
		}
	} // namespace

	void AMIPSEnergy::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		if (params.contains("use_rest_pose"))
		{
			use_rest_pose_ = params["use_rest_pose"].get<bool>();
		}

		if (energy_weights_.size() <= index)
			energy_weights_.resize(index + 1, 1.0);

		if (params.contains("weight"))
			energy_weights_[index] = params["weight"].get<double>();
		else
			energy_weights_[index] = 1.0;

		json power_params = params;
		if (!power_params.contains("power"))
			power_params["power"] = 1.0;
		power_.add_multimaterial(index, power_params, "", root_path);
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
		const double power = power_(p, t, el_id);

		if (power == 1.0)
		{
			if (use_rest_pose_)
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
		else if(power == 2.0)
		{
			//TODO: add autogen for amips squared gradient
		}
		return autodiff_gradient(*this, p, t, el_id, F);
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
		const double power = power_(p, t, el_id);

		if (power == 1.0)
		{
			if (use_rest_pose_)
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
		else if(power == 2.0)
		{
			//TODO: add autogen for amips squared hessian
		}

		return autodiff_hessian(*this, p, t, el_id, F);
	}

} // namespace polyfem::assembler