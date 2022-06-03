#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/ElasticityUtils.hpp>

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElementBases.hpp>
#include <polyfem/AutodiffTypes.hpp>
#include <polyfem/Types.hpp>

#include <Eigen/Dense>
#include <array>

//attempt at Ogden, incomplete, not used, and not working
namespace polyfem
{
	namespace assembler
	{
		class OgdenElasticity
		{
		public:
			OgdenElasticity();

			Eigen::MatrixXd assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
			Eigen::VectorXd assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
			double compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
			compute_rhs(const AutodiffHessianPt &pt) const;

			inline int size() const { return size_; }
			void set_size(const int size);

			void set_stiffness_tensor(int i, int j, const double val);
			double stifness_tensor(int i, int j) const;

			void compute_von_mises_stresses(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
			void compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

			void set_parameters(const json &params);

		private:
			int size_ = 2;

			Eigen::VectorXd alphas_;
			Eigen::VectorXd mus_;
			Eigen::VectorXd Ds_;

			template <typename T>
			T compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

			void assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
		};
	} // namespace assembler
} // namespace polyfem
