#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <cppoptlib/problem.h>

#include <Eigen/Dense>
#include <array>

// non linear NeoHookean material model
namespace polyfem
{
	class State;

	namespace assembler
	{
		class Multiscale : public NLAssembler, public ElasticityAssembler
		{
		public:
			Multiscale();
			Multiscale(std::shared_ptr<State> state_ptr);
			virtual ~Multiscale();

			using NLAssembler::assemble_energy;
			using NLAssembler::assemble_gradient;
			using NLAssembler::assemble_hessian;

			// energy, gradient, and hessian used in newton method
			double compute_energy(const NonLinearAssemblerData &data) const override;
			Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
			virtual Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

			void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

			// sets material params
			virtual void add_multimaterial(const int index, const json &params) override;

			void test_reduced_basis(const std::vector<Eigen::MatrixXd> &def_grads, Eigen::VectorXd &energy_errors, Eigen::VectorXd &stress_errors);

			std::shared_ptr<State> get_microstructure_state() { return state; }

			void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

			virtual void homogenization(const Eigen::MatrixXd &def_grad, double &energy) const;
			virtual void homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress) const;
			virtual void homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress, Eigen::MatrixXd &stiffness) const;

			double homogenize_energy(const Eigen::MatrixXd &x) const;
			Eigen::MatrixXd homogenize_def_grad(const Eigen::MatrixXd &x) const;
			void homogenize_stress(const Eigen::MatrixXd &x, Eigen::MatrixXd &stress) const;
			virtual void homogenize_stiffness(const Eigen::MatrixXd &x, Eigen::MatrixXd &stiffness) const;

			std::map<std::string, ParamFunc> parameters() const override;
			virtual std::string name() const override { return "Multiscale"; }

		protected:
			std::shared_ptr<polyfem::State> state;
			double microstructure_volume = 0;
		};
	} // namespace assembler
} // namespace polyfem
