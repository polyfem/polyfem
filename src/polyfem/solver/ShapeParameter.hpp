#pragma once

#include "Parameter.hpp"

#include <ipc/collisions/collision_constraint.hpp>
#include <ipc/broad_phase/broad_phase.hpp>
#include "constraints/ShapeConstraints.hpp"

namespace polyfem
{
	class ShapeParameter : public Parameter
	{
	public:
		ShapeParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override
		{
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			states_ptr_[0]->get_vf(V, F);
			Eigen::VectorXd reduced;
			shape_constraints_->full_to_reduced(V, reduced);
			return reduced;
		}

		Eigen::MatrixXd map(const Eigen::VectorXd &x) const;
		Eigen::VectorXd map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const override;

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end() override;
		void post_step(const int iter_num, const Eigen::VectorXd &x0) override;

		bool pre_solve(const Eigen::VectorXd &newX) override;
		void post_solve(const Eigen::VectorXd &newX) override;

		bool remesh(Eigen::VectorXd &x) override;
		void build_active_nodes();
		void build_tied_nodes(); // not applied to shape constraints

		std::vector<int> get_constrained_nodes() const;
		inline void get_updated_nodes(const Eigen::VectorXd &x, const Eigen::MatrixXd &new_V_rest, Eigen::MatrixXd &V) const { return shape_constraints_->reduced_to_full(x, new_V_rest, V); }

		static bool internal_smoothing(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const std::vector<int> &boundary_indices, const Eigen::MatrixXd &boundary_constraints, const json &slim_params, Eigen::MatrixXd &smooth_field);
		static bool is_flipped(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

		std::map<int, std::vector<int>> optimization_boundary_to_node;

		const json &get_shape_params() const { return shape_params; }

	private:
		int iter = 0;
		int dim;


		std::vector<bool> active_nodes_mask;
		std::vector<bool> free_dimension; // not applied to shape constraints

		std::vector<bool> tied_nodes_mask;          // not applied to shape constraints
		std::vector<std::array<int, 2>> tied_nodes; // not applied to shape constraints

		bool mesh_flipped = false;

		json shape_params, slim_params;

		// below only used for problems with contact
		// TODO: move to objective

		bool has_collision;

		ipc::BroadPhaseMethod _broad_phase_method;
		double _ccd_tolerance;
		int _ccd_max_iterations;

		ipc::CollisionMesh collision_mesh;

		std::unique_ptr<ShapeConstraints> shape_constraints_;
	};
} // namespace polyfem