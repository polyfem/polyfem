#pragma once

#include "AssemblerData.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Types.hpp>

#include <cppoptlib/problem.h>

#include <Eigen/Dense>
#include <array>

// non linear NeoHookean material model
namespace polyfem
{
	class State;

	namespace assembler
	{
		class Multiscale
		{
		public:
			Multiscale();
			virtual ~Multiscale();

			// energy, gradient, and hessian used in newton method
			virtual Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const;
			Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const;

			double compute_energy(const NonLinearAssemblerData &data) const;

			// rhs for fabbricated solution, compute with automatic sympy code
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
			compute_rhs(const AutodiffHessianPt &pt) const;

			inline int size() const { return size_; }
			void set_size(const int size);

			// von mises and stress tensor
			void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
			void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

			// sets material params
			virtual void add_multimaterial(const int index, const json &params);
			void set_params(const LameParameters &params) { params_ = params; }
			LameParameters &lame_params() { return params_; }
			const LameParameters &lame_params() const { return params_; }

			void test_reduced_basis(const std::vector<Eigen::MatrixXd> &def_grads, Eigen::VectorXd &energy_errors, Eigen::VectorXd &stress_errors);

			std::shared_ptr<State> get_microstructure_state() { return state; }

			void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
			
			virtual void homogenization(const Eigen::MatrixXd &def_grad, double &energy) const;
			virtual void homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress) const;

			double homogenize_energy(const Eigen::MatrixXd &x) const;
			void homogenize_stress(const Eigen::MatrixXd &x, Eigen::MatrixXd &stress) const;

			virtual std::string name() const { return "Multiscale"; }

		protected:
			int size_ = -1;

			LameParameters params_;

			std::shared_ptr<polyfem::State> state;
			double microstructure_volume = 0;
		};
	} // namespace assembler
} // namespace polyfem
