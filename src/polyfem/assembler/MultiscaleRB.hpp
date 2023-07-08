#pragma once

#include "AssemblerData.hpp"
#include "Multiscale.hpp"

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
		class MultiscaleRB : public Multiscale
		{
		public:
			MultiscaleRB();
			~MultiscaleRB();

			using NLAssembler::assemble_energy;
			using NLAssembler::assemble_gradient;
			using NLAssembler::assemble_hessian;
			
			// energy, gradient, and hessian used in newton method
			Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

			// sets material params
			void add_multimaterial(const int index, const json &params) override;

			void test_reduced_basis(const std::vector<Eigen::MatrixXd> &def_grads, Eigen::VectorXd &energy_errors, Eigen::VectorXd &stress_errors);

		private:
			// microstructure
			Eigen::MatrixXd reduced_basis; // (n_bases*dim) * N
			void sample_def_grads(const Eigen::VectorXd &sample_det, const Eigen::VectorXd &sample_amp, const int n_sample_dir, std::vector<Eigen::MatrixXd> &def_grads) const;
			void create_reduced_basis(const std::vector<Eigen::MatrixXd> &def_grads);
			void projection(const Eigen::MatrixXd &F, Eigen::MatrixXd &x) const;

			void homogenization(const Eigen::MatrixXd &def_grad, double &energy) const override;
			void homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress) const override;
			void homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress, Eigen::MatrixXd &stiffness) const override;

			void homogenize_stiffness(const Eigen::MatrixXd &x, Eigen::MatrixXd &stiffness) const override;

			std::string name() const override { return "MultiscaleRB"; }

			int n_reduced_basis = 5;
		};

		class MultiscaleRBProblem : public cppoptlib::Problem<double>
		{
		public:
			using typename cppoptlib::Problem<double>::Scalar;
			using typename cppoptlib::Problem<double>::TVector;
			typedef StiffnessMatrix THessian;

			MultiscaleRBProblem(const std::shared_ptr<State> &state_ptr, const Eigen::MatrixXd &reduced_basis);
			~MultiscaleRBProblem() = default;

			void set_linear_disp(const Eigen::MatrixXd &linear_sol) { linear_sol_ = linear_sol; }

			double value(const TVector &x) override;
			void gradient(const TVector &x, TVector &gradv) override;
			void hessian(const TVector &x, THessian &hessian);
			void hessian(const TVector &x, Eigen::MatrixXd &hessian) override;

			TVector component_values(const TVector &x)
			{
				TVector val(1);
				val(0) = value(x);
				return val;
			}

			Eigen::MatrixXd component_gradients(const TVector &x)
			{
				Eigen::VectorXd grad(x.size());
				gradient(x, grad);
				return grad;
			}

			Eigen::MatrixXd coeff_to_field(const TVector &x)
			{
				return linear_sol_ + reduced_basis_ * x;
			}

			bool is_step_valid(const TVector &x0, const TVector &x1)
			{
				TVector gradv;
				gradient(x1, gradv);
				if (std::isnan(gradv.norm()))
					return false;
				return true;
			}

			void set_project_to_psd(bool val) {}
			void save_to_file(const TVector &x0) {}
			void solution_changed(const TVector &newX) {}
			void line_search_begin(const TVector &x0, const TVector &x1) {}
			void line_search_end() {}
			void post_step(const int iter_num, const TVector &x) {}
			bool smoothing(const TVector &x, const TVector &new_x, TVector &smoothed_x) { return false; }
			bool stop(const TVector &x) { return false; }
			bool remesh(TVector &x) { return false; }
			double max_step_size(const TVector &x0, const TVector &x1) { return 1; }
			bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; }

		private:
			std::shared_ptr<const State> state;
			double microstructure_volume = 0;

			Eigen::MatrixXd linear_sol_;
			const Eigen::MatrixXd &reduced_basis_;
		};
	} // namespace assembler
} // namespace polyfem
