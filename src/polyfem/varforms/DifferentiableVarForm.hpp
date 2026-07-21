#pragma once

#include <polyfem/varforms/VarForm.hpp>

namespace polyfem::varform
{
	class DifferentiableVarForm : public VarForm
	{
	public:
		~DifferentiableVarForm() override = default;

		virtual const FESpace &primary_space() const = 0;
		virtual const VarFormBoundaryState &boundary_state() const = 0;
		virtual const assembler::Assembler &primary_assembler() const = 0;
		virtual const assembler::Mass &mass_assembler() const = 0;
		virtual const assembler::AssemblyValsCache &assembly_cache() const = 0;
		virtual const assembler::AssemblyValsCache &mass_assembly_cache() const = 0;
		virtual const StiffnessMatrix &mass_matrix() const = 0;
		virtual solver::SolveData *solve_data() { return nullptr; }
		virtual const solver::SolveData *solve_data() const { return nullptr; }
		virtual const ipc::CollisionMesh &collision_mesh() const;
		virtual const mesh::Obstacle &get_obstacle() const;
		virtual const assembler::ViscousDamping *damping_assembler() const { return nullptr; }
		virtual const assembler::ViscousDampingPrev *damping_prev_assembler() const { return nullptr; }
		virtual void initial_solution(Eigen::MatrixXd &solution, const InitialConditionOverride *override = nullptr) const;
		virtual void initial_velocity(Eigen::MatrixXd &velocity, const InitialConditionOverride *override = nullptr) const;
		virtual void initial_acceleration(Eigen::MatrixXd &acceleration, const InitialConditionOverride *override = nullptr) const;

		void build_stiffness_matrix(StiffnessMatrix &stiffness) const;
		std::vector<int> primitive_to_node() const;
		std::vector<int> node_to_primitive() const;
		void get_elements(Eigen::MatrixXi &elements) const;
		QuadratureOrders n_boundary_samples() const;
		bool is_problem_linear() const;

		virtual void set_lame_parameters(const Eigen::VectorXd &lambda, const Eigen::VectorXd &mu);
		virtual void set_friction_coefficient(double coefficient);
		virtual void set_damping_coefficients(double psi, double phi);
		virtual void set_dirichlet_boundary(int boundary_id, int time_step, const Eigen::VectorXd &value);
		virtual void set_dirichlet_nodes(const Eigen::VectorXi &input_nodes, const Eigen::MatrixXd &values);
		virtual void set_pressure_boundary(int boundary_id, int time_step, double value);
	};
} // namespace polyfem::varform
