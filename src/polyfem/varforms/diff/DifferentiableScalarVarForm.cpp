#include <polyfem/varforms/diff/DifferentiableScalarVarForm.hpp>

namespace polyfem::varform
{
	std::string DifferentiableScalarVarForm::name() const
	{
		return ScalarVarForm::name();
	}

	void DifferentiableScalarVarForm::solve(
		Eigen::MatrixXd &solution,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step,
		const bool)
	{
		ScalarVarForm::solve(solution, initial_condition_override, post_step);
	}

	void DifferentiableScalarVarForm::prepare()
	{
		ScalarVarForm::prepare();
	}

	void DifferentiableScalarVarForm::save_vtu(
		const std::string &path,
		const Eigen::MatrixXd &solution,
		const double time,
		const double dt) const
	{
		const io::OutputSpace space = ScalarVarForm::output_space();
		ScalarVarForm::ensure_output_sampler();
		const auto opts = ScalarVarForm::export_options(space);
		output_geometry_.save_vtu(
			path, space, ScalarVarForm::output_field_function(solution, opts), time, dt, opts);
	}

	json &DifferentiableScalarVarForm::get_args() { return args; }
	const json &DifferentiableScalarVarForm::get_args() const { return args; }

	const mesh::Mesh &DifferentiableScalarVarForm::get_mesh() const
	{
		assert(mesh_ && "The mesh must be loaded before it is accessed");
		return *mesh_;
	}

	assembler::Problem &DifferentiableScalarVarForm::get_problem()
	{
		assert(problem && "The problem must be initialized before it is accessed");
		return *problem;
	}

	const assembler::Problem &DifferentiableScalarVarForm::get_problem() const
	{
		assert(problem && "The problem must be initialized before it is accessed");
		return *problem;
	}

	const std::string &DifferentiableScalarVarForm::get_root_path() const { return root_path; }

	std::string DifferentiableScalarVarForm::input_path(const std::string &path, const bool only_if_exists) const
	{
		return ScalarVarForm::resolve_input_path(path, only_if_exists);
	}

	std::string DifferentiableScalarVarForm::output_file_path(const std::string &path) const
	{
		return ScalarVarForm::resolve_output_path(path);
	}

	const Units &DifferentiableScalarVarForm::get_units() const { return units; }
	bool DifferentiableScalarVarForm::is_contact_enabled() const { return ScalarVarForm::is_contact_enabled(); }

	const FESpace &DifferentiableScalarVarForm::primary_space() const { return space_; }
	const VarFormBoundaryState &DifferentiableScalarVarForm::boundary_state() const { return boundary_; }

	const assembler::Assembler &DifferentiableScalarVarForm::primary_assembler() const
	{
		assert(primary_assembler_ && "The primary assembler must be initialized before it is accessed");
		return *primary_assembler_;
	}

	const assembler::Mass &DifferentiableScalarVarForm::mass_assembler() const
	{
		assert(mass_assembler_ && "The mass assembler must be initialized before it is accessed");
		return *mass_assembler_;
	}

	const assembler::AssemblyValsCache &DifferentiableScalarVarForm::assembly_cache() const { return ass_vals_cache_; }
	const assembler::AssemblyValsCache &DifferentiableScalarVarForm::mass_assembly_cache() const { return mass_ass_vals_cache_; }
	const StiffnessMatrix &DifferentiableScalarVarForm::mass_matrix() const { return mass_; }
	solver::SolveData *DifferentiableScalarVarForm::solve_data() { return &solve_data_; }
	const solver::SolveData *DifferentiableScalarVarForm::solve_data() const { return &solve_data_; }

	mesh::Mesh &DifferentiableScalarVarForm::mutable_mesh()
	{
		assert(mesh_ && "Vertex positions can only be updated after loading a mesh");
		return *mesh_;
	}

	void DifferentiableScalarVarForm::invalidate_after_geometry_update()
	{
		time_integrator = nullptr;
		rhs_assembler_ = nullptr;
		solve_data_ = solver::SolveData();
		prepared_ = false;
		output_sampler_initialized_ = false;
	}

	void DifferentiableScalarVarForm::invalidate_after_parameter_update()
	{
		time_integrator = nullptr;
		rhs_assembler_ = nullptr;
		solve_data_ = solver::SolveData();
		prepared_ = false;
	}

	QuadratureOrders DifferentiableScalarVarForm::boundary_samples(
		const int discr_order,
		const int discr_orderq,
		const int geometry_discr_order) const
	{
		return ScalarVarForm::n_boundary_samples(discr_order, discr_orderq, geometry_discr_order);
	}
} // namespace polyfem::varform
