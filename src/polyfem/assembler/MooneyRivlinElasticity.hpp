#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <array>

//non linear NeoHookean material model
namespace polyfem
{
	namespace assembler
	{
		class MooneyRivlinElasticity
		{
		public:
			MooneyRivlinElasticity();

			//energy, gradient, and hessian used in newton method
			Eigen::MatrixXd assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
			Eigen::VectorXd assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

			double compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

			//rhs for fabbricated solution, compute with automatic sympy code
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
			compute_rhs(const AutodiffHessianPt &pt) const;

			inline int size() const { return size_; }
			void set_size(const int size);

			//von mises and stress tensor
			void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
			void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

			//sets material params
			void add_multimaterial(const int index, const json &params);
			// void set_params(const LameParameters &params) { params_ = params; }

		private:
			int size_ = -1;

			double c1_ = 0; 
			double c2_ = 0;
			double k_ = 0;

			//utulity function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
			template <typename T>
			T compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
			void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
		};
	} // namespace assembler
} // namespace polyfem