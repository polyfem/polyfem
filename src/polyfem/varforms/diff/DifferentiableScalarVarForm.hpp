#pragma once

#include <polyfem/varforms/ScalarVarForm.hpp>
#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>

namespace polyfem::varform
{
	class DifferentiableScalarVarForm final : public ScalarVarForm, public DifferentiableVarForm
	{
	public:
		std::string name() const override;
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

	protected:
		mesh::Mesh &mutable_mesh() override;
		void invalidate_after_geometry_update() override;
		void invalidate_after_parameter_update() override;
		QuadratureOrders boundary_samples(int discr_order, int discr_orderq, int geometry_discr_order) const override;
	};
} // namespace polyfem::varform
