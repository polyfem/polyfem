#pragma once

#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/solver/SolveData.hpp>

#include <ipc/collision_mesh.hpp>

#include <array>
#include <unordered_map>

namespace polyfem::mesh
{
	class Obstacle;
}

namespace polyfem::varform
{
	/// Optimization-facing interface implemented by differentiated VarForm adapters.
	class DifferentiableVarForm
	{
	public:
		virtual ~DifferentiableVarForm() = default;

		virtual std::string name() const = 0;
		virtual void solve(
			Eigen::MatrixXd &solution,
			const InitialConditionOverride *initial_condition_override,
			const ForwardStepCallback &post_step,
			bool differentiable) = 0;
		virtual void prepare() = 0;
		virtual void save_vtu(const std::string &path, const Eigen::MatrixXd &solution, double time, double dt) const = 0;

		virtual json &get_args() = 0;
		virtual const json &get_args() const = 0;
		virtual const mesh::Mesh &get_mesh() const = 0;
		virtual assembler::Problem &get_problem() = 0;
		virtual const assembler::Problem &get_problem() const = 0;
		virtual const std::string &get_root_path() const = 0;
		virtual std::string input_path(const std::string &path, bool only_if_exists = false) const = 0;
		virtual std::string output_file_path(const std::string &path) const = 0;
		virtual const Units &get_units() const = 0;
		virtual bool is_contact_enabled() const = 0;

		virtual const FESpace &primary_space() const = 0;
		virtual const VarFormBoundaryState &boundary_state() const = 0;
		virtual const assembler::Assembler &primary_assembler() const = 0;
		virtual const assembler::Mass &mass_assembler() const = 0;
		virtual const assembler::AssemblyValsCache &assembly_cache() const = 0;
		virtual const assembler::AssemblyValsCache &mass_assembly_cache() const = 0;
		virtual const StiffnessMatrix &mass_matrix() const = 0;
		virtual solver::SolveData *solve_data() = 0;
		virtual const solver::SolveData *solve_data() const = 0;
		virtual const ipc::CollisionMesh &collision_mesh() const;
		virtual const mesh::Obstacle &get_obstacle() const;
		virtual const assembler::ViscousDamping *damping_assembler() const { return nullptr; }
		virtual const assembler::ViscousDampingPrev *damping_prev_assembler() const { return nullptr; }
		virtual void initial_solution(Eigen::MatrixXd &solution, const InitialConditionOverride *override = nullptr) const;
		virtual void initial_velocity(Eigen::MatrixXd &velocity, const InitialConditionOverride *override = nullptr) const;
		virtual void initial_acceleration(Eigen::MatrixXd &acceleration, const InitialConditionOverride *override = nullptr) const;
		virtual Eigen::MatrixXd displacement_gradient() const;

		void get_vertices(Eigen::MatrixXd &vertices) const;
		std::unordered_map<int, std::array<bool, 3>> boundary_conditions_ids(const std::string &bc_type) const;
		bool is_homogenization() const;
		bool has_periodic_boundary() const;
		Eigen::MatrixXd periodic_tile_offsets() const;
		bool is_adhesion_enabled() const;
		bool is_pressure_enabled() const;
		bool has_constraints() const;
		bool is_problem_linear() const;

		void build_stiffness_matrix(StiffnessMatrix &stiffness) const;
		std::vector<int> primitive_to_node() const;
		std::vector<int> node_to_primitive() const;
		void get_elements(Eigen::MatrixXi &elements) const;
		QuadratureOrders n_boundary_samples() const;

		void set_vertex_positions(const Eigen::MatrixXd &vertices);
		virtual void set_lame_parameters(const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu);
		virtual void set_friction_coefficient(double coefficient);
		virtual void set_damping_coefficients(double psi, double phi);
		virtual void set_dirichlet_boundary(int boundary_id, int time_step, const Eigen::VectorXd &value);
		virtual void set_dirichlet_nodes(const Eigen::VectorXi &input_nodes, const Eigen::MatrixXd &values);
		virtual void set_pressure_boundary(int boundary_id, int time_step, double value);

	protected:
		virtual mesh::Mesh &mutable_mesh() = 0;
		virtual void invalidate_after_geometry_update() = 0;
		virtual void invalidate_after_parameter_update() = 0;
		virtual QuadratureOrders boundary_samples(int discr_order, int discr_orderq, int geometry_discr_order) const = 0;
	};
} // namespace polyfem::varform
