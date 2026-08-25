#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
#include <iostream>
#include <vector>

namespace polyfem::assembler
{
	class AMIPSEnergy : public GenericElastic<AMIPSEnergy>
	{
	public:
		AMIPSEnergy()
		{
			autodiff_type_ = AutodiffType::NONE;
		}

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		std::string name() const override { return "AMIPS"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

		bool allow_inversion() const override { return false; }

		bool real_def_grad() const override { return use_rest_pose_; }

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			DefGradMatrix<T> &def_grad) const
		{
			typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

			double power = -1;
			if (use_rest_pose_)
				power = size() == 2 ? 1. : (2. / 3.);
			else
				power = size() == 2 ? 2. : 5. / 3.;

			AutoDiffGradMat standard;

			if (size() == 2)
				standard = get_standard<2, T>(size(), use_rest_pose_);
			else
				standard = get_standard<3, T>(size(), use_rest_pose_);

			if (!use_rest_pose_)
				def_grad = def_grad * standard;

			const T det = polyfem::utils::determinant(def_grad);
			if (det <= 0)
			{
				return T(std::nan(""));
			}

			const T powJ = pow(det, power);
			const double weight = get_energy_weight(el_id);
			return T(weight) * (def_grad.transpose() * def_grad).trace() / powJ; //+ barrier<T>::value(det);
		}

	private:
		double get_energy_weight(const int el_id) const;
		std::vector<double> energy_weights_;
		bool use_rest_pose_ = false;

		template <int dimt, class T>
		static Eigen::Matrix<T, dimt, dimt> get_standard(const int dim, const bool use_rest_pose)
		{
			Eigen::Matrix<double, dimt, dimt> standard(dim, dim);
			if (use_rest_pose)
			{
				standard.setIdentity();
			}
			else
			{
				if (dim == 2)
					standard << 1, 0,
						0.5, std::sqrt(3) / 2;
				else
					standard << 1, 0, 0,
						0.5, std::sqrt(3) / 2., 0,
						0.5, 0.5 / std::sqrt(3), std::sqrt(3) / 2.;
				standard = standard.inverse().transpose().eval();
			}

			Eigen::Matrix<T, dimt, dimt> res(dim, dim);
			for (int i = 0; i < dim; ++i)
			{
				for (int j = 0; j < dim; ++j)
				{
					res(i, j) = T(standard(i, j));
				}
			}

			return res;
		}

	public:
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> gradient(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const override;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hessian(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const override;
	};

} // namespace polyfem::assembler