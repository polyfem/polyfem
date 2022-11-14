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
		class MultiscaleRB
		{
		public:
			MultiscaleRB();
			~MultiscaleRB();

			// energy, gradient, and hessian used in newton method
			Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const;
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
			void add_multimaterial(const int index, const json &params);
			void set_params(const LameParameters &params) { params_ = params; }
			LameParameters &lame_params() { return params_; }
			const LameParameters &lame_params() const { return params_; }

			void test_reduced_basis(const std::vector<Eigen::MatrixXd> &def_grads, Eigen::VectorXd &energy_errors, Eigen::VectorXd &stress_errors);

		private:
			int size_ = -1;

			LameParameters params_;

			// utulity function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
			double compute_energy_aux(const NonLinearAssemblerData &data) const;
			template <int n_basis, int dim>
			void compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const;
			template <int n_basis, int dim>
			void compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::VectorXd &G_flattened) const;

			void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;

			// microstructure
			json unit_cell_args;
			Eigen::MatrixXd reduced_basis; // (n_bases*dim) * N
			void sample_def_grads(const Eigen::VectorXd &sample_det, const Eigen::VectorXd &sample_amp, const int n_sample_dir, std::vector<Eigen::MatrixXd> &def_grads) const;
			void create_reduced_basis(const std::vector<Eigen::MatrixXd> &def_grads);
			void projection(const Eigen::MatrixXd &F, Eigen::MatrixXd &x) const;
			
			void homogenization(const Eigen::MatrixXd &def_grad, double &energy) const;
			void homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress) const;
			void homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress, Eigen::MatrixXd &stiffness) const;
			void brute_force_homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress, Eigen::MatrixXd &fluctuated) const;

			double homogenize_energy(const Eigen::MatrixXd &x) const;
			void homogenize_stress(const Eigen::MatrixXd &x, Eigen::MatrixXd &stress) const;
			void homogenize_stiffness(const Eigen::MatrixXd &x, Eigen::MatrixXd &stiffness) const;

			int n_reduced_basis = 5;

			std::shared_ptr<polyfem::State> state;
			double microstructure_volume = 0;
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

			double value(const TVector &x) { return value(x, false); }
			double value(const TVector &x, const bool only_elastic);

			double target_value(const TVector &x) { return value(x); }
			void gradient(const TVector &x, TVector &gradv) { gradient(x, gradv, false); }
			void gradient(const TVector &x, TVector &gradv, const bool only_elastic);
			void target_gradient(const TVector &x, TVector &gradv) { gradient(x, gradv); }
			void hessian(const TVector &x, THessian &hessian);
			void hessian(const TVector &x, Eigen::MatrixXd &hessian);

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

			bool verify_gradient(const TVector &x, const TVector &gradv) { return true; }
			
			void set_project_to_psd(bool val) {}
			void save_to_file(const TVector &x0) {}
			void solution_changed(const TVector &newX) {}
			void line_search_begin(const TVector &x0, const TVector &x1) {}
			void line_search_end() {}
			void post_step(const int iter_num, const TVector &x) {}
			void smoothing(const TVector &x, TVector &new_x) {}
			bool is_intersection_free(const TVector &x) { return true; }
			bool stop(const TVector &x) { return false; }
			bool remesh(TVector &x) { return false; }
			TVector force_inequality_constraint(const TVector &x0, const TVector &dx) { return x0 + dx; }
			double max_step_size(const TVector &x0, const TVector &x1) { return 1; }
			bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; }
			int n_inequality_constraints() { return 0; }
			double inequality_constraint_val(const TVector &x, const int index)
			{
				assert(false);
				return std::nan("");
			}
			TVector inequality_constraint_grad(const TVector &x, const int index)
			{
				assert(false);
				return TVector();
			}

		private:
			std::shared_ptr<const State> state;
			double microstructure_volume = 0;
			
			Eigen::MatrixXd linear_sol_;
			const Eigen::MatrixXd &reduced_basis_;
		};
	} // namespace assembler
} // namespace polyfem
