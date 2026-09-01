#include <polyfem/varforms/diff/DifferentiableLinearElasticVarForm.hpp>

namespace polyfem::varform
{
	std::string DifferentiableLinearElasticVarForm::name() const
	{
		return LinearElasticVarForm::name();
	}

	void DifferentiableLinearElasticVarForm::solve(
		Eigen::MatrixXd &solution,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step,
		const bool)
	{
		LinearElasticVarForm::solve(solution, initial_condition_override, post_step);
	}

	void DifferentiableLinearElasticVarForm::prepare()
	{
		LinearElasticVarForm::prepare();
	}

	void DifferentiableLinearElasticVarForm::save_vtu(
		const std::string &path,
		const Eigen::MatrixXd &solution,
		const double time,
		const double dt) const
	{
		const io::OutputSpace space = LinearElasticVarForm::output_space();
		LinearElasticVarForm::ensure_output_sampler();
		const auto opts = LinearElasticVarForm::export_options(space);
		output_geometry_.save_vtu(
			path, space, LinearElasticVarForm::output_field_function(solution, opts), time, dt, opts);
	}

	json &DifferentiableLinearElasticVarForm::get_args() { return args; }
	const json &DifferentiableLinearElasticVarForm::get_args() const { return args; }

	const mesh::Mesh &DifferentiableLinearElasticVarForm::get_mesh() const
	{
		assert(mesh_ && "The mesh must be loaded before it is accessed");
		return *mesh_;
	}

	assembler::Problem &DifferentiableLinearElasticVarForm::get_problem()
	{
		assert(problem && "The problem must be initialized before it is accessed");
		return *problem;
	}

	const assembler::Problem &DifferentiableLinearElasticVarForm::get_problem() const
	{
		assert(problem && "The problem must be initialized before it is accessed");
		return *problem;
	}

	const std::string &DifferentiableLinearElasticVarForm::get_root_path() const { return root_path; }

	std::string DifferentiableLinearElasticVarForm::input_path(const std::string &path, const bool only_if_exists) const
	{
		return LinearElasticVarForm::resolve_input_path(path, only_if_exists);
	}

	std::string DifferentiableLinearElasticVarForm::output_file_path(const std::string &path) const
	{
		return LinearElasticVarForm::resolve_output_path(path);
	}

	const Units &DifferentiableLinearElasticVarForm::get_units() const { return units; }
	bool DifferentiableLinearElasticVarForm::is_contact_enabled() const { return LinearElasticVarForm::is_contact_enabled(); }

	const FESpace &DifferentiableLinearElasticVarForm::primary_space() const { return space_; }
	const VarFormBoundaryState &DifferentiableLinearElasticVarForm::boundary_state() const { return boundary_; }

	const assembler::Assembler &DifferentiableLinearElasticVarForm::primary_assembler() const
	{
		assert(primary_assembler_ && "The primary assembler must be initialized before it is accessed");
		return *primary_assembler_;
	}

	const assembler::Mass &DifferentiableLinearElasticVarForm::mass_assembler() const
	{
		assert(mass_assembler_ && "The mass assembler must be initialized before it is accessed");
		return *mass_assembler_;
	}

	const assembler::AssemblyValsCache &DifferentiableLinearElasticVarForm::assembly_cache() const { return ass_vals_cache_; }
	const assembler::AssemblyValsCache &DifferentiableLinearElasticVarForm::mass_assembly_cache() const { return mass_ass_vals_cache_; }
	const StiffnessMatrix &DifferentiableLinearElasticVarForm::mass_matrix() const { return mass_; }
	solver::SolveData *DifferentiableLinearElasticVarForm::solve_data() { return &solve_data_; }
	const solver::SolveData *DifferentiableLinearElasticVarForm::solve_data() const { return &solve_data_; }

	void DifferentiableLinearElasticVarForm::initial_solution(
		Eigen::MatrixXd &solution,
		const InitialConditionOverride *override) const
	{
		LinearElasticVarForm::initial_solution(solution, override);
	}

	void DifferentiableLinearElasticVarForm::initial_velocity(
		Eigen::MatrixXd &velocity,
		const InitialConditionOverride *override) const
	{
		LinearElasticVarForm::initial_velocity(velocity, override);
	}

	void DifferentiableLinearElasticVarForm::initial_acceleration(
		Eigen::MatrixXd &acceleration,
		const InitialConditionOverride *override) const
	{
		LinearElasticVarForm::initial_acceleration(acceleration, override);
	}

	mesh::Mesh &DifferentiableLinearElasticVarForm::mutable_mesh()
	{
		assert(mesh_ && "Vertex positions can only be updated after loading a mesh");
		return *mesh_;
	}

	void DifferentiableLinearElasticVarForm::invalidate_after_geometry_update()
	{
		solve_data_.elastic_form = nullptr;
		solve_data_.body_form = nullptr;
		solve_data_.inertia_form = nullptr;
		solve_data_.time_integrator = nullptr;
		rhs_assembler_ = nullptr;
		prepared_ = false;
		output_sampler_initialized_ = false;
	}

	void DifferentiableLinearElasticVarForm::invalidate_after_parameter_update()
	{
		solve_data_.elastic_form = nullptr;
		solve_data_.body_form = nullptr;
		solve_data_.inertia_form = nullptr;
		solve_data_.time_integrator = nullptr;
		rhs_assembler_ = nullptr;
		prepared_ = false;
	}

	QuadratureOrders DifferentiableLinearElasticVarForm::boundary_samples(
		const int discr_order,
		const int discr_orderq,
		const int geometry_discr_order) const
	{
		return LinearElasticVarForm::n_boundary_samples(discr_order, discr_orderq, geometry_discr_order);
	}
} // namespace polyfem::varform
