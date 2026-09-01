#pragma once

#include <polyfem/varforms/NonlinearElasticVarForm.hpp>
#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>
#include <polyfem/assembler/MacroStrain.hpp>

namespace polyfem::varform
{
	class DifferentiableNonlinearElasticVarForm : public NonlinearElasticVarForm, public DifferentiableVarForm
	{
	public:
		void solve(
			Eigen::MatrixXd &solution,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step,
			bool differentiable) override;
		void prepare() override;
		void save_vtu(const std::string &path, const Eigen::MatrixXd &solution, double time, double dt) const override;

		json &get_args() override;
		const json &get_args() const override;
		const mesh::Mesh &get_mesh() const override;
		assembler::Problem &get_problem() override;
		const assembler::Problem &get_problem() const override;
		const std::string &get_root_path() const override;
		std::string input_path(const std::string &path, bool only_if_exists = false) const override;
		std::string output_file_path(const std::string &path) const override;
		const Units &get_units() const override;
		bool is_contact_enabled() const override;

		const FESpace &primary_space() const override;
		const VarFormBoundaryState &boundary_state() const override;
		const assembler::Assembler &primary_assembler() const override;
		const assembler::Mass &mass_assembler() const override;
		const assembler::AssemblyValsCache &assembly_cache() const override;
		const assembler::AssemblyValsCache &mass_assembly_cache() const override;
		const StiffnessMatrix &mass_matrix() const override;
		solver::SolveData *solve_data() override;
		const solver::SolveData *solve_data() const override;
		const ipc::CollisionMesh &collision_mesh() const override;
		const mesh::Obstacle &get_obstacle() const override;
		const assembler::ViscousDamping *damping_assembler() const override;
		const assembler::ViscousDampingPrev *damping_prev_assembler() const override;
		void initial_solution(Eigen::MatrixXd &solution, const InitialConditionOverride *override = nullptr) const override;
		void initial_velocity(Eigen::MatrixXd &velocity, const InitialConditionOverride *override = nullptr) const override;
		void initial_acceleration(Eigen::MatrixXd &acceleration, const InitialConditionOverride *override = nullptr) const override;
		Eigen::MatrixXd displacement_gradient() const override;

	protected:
		mesh::Mesh &mutable_mesh() override;
		void invalidate_after_geometry_update() override;
		void invalidate_after_parameter_update() override;
		QuadratureOrders boundary_samples(int discr_order, int discr_orderq, int geometry_discr_order) const override;
		void init_forms(const json &args, int dim, Eigen::MatrixXd &solution, double time) override;
		void solve_tensor_nonlinear(int step, Eigen::MatrixXd &solution, bool init_lagging = true) override;

	protected:
		void init_homogenization_solve(
			Eigen::MatrixXd &solution,
			double time,
			const InitialConditionOverride *initial_condition_override);
		void solve_homogenization_step(Eigen::MatrixXd &solution, const ForwardStepCallback &post_step);

	private:
		bool differentiable_mode_ = false;
		assembler::MacroStrainValue macro_strain_constraint_;
		Eigen::MatrixXd displacement_gradient_;
	};

	class DifferentiableNonlinearElasticStaticVarForm final : public DifferentiableNonlinearElasticVarForm
	{
	public:
		std::string name() const override { return "NonlinearElasticStatic"; }

	private:
		void solve_problem(
			Eigen::MatrixXd &solution,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step) override;
	};

	class DifferentiableNonlinearElasticTransientVarForm final : public DifferentiableNonlinearElasticVarForm
	{
	public:
		std::string name() const override { return "NonlinearElasticTransient"; }

	private:
		void solve_problem(
			Eigen::MatrixXd &solution,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step) override;
	};
} // namespace polyfem::varform
