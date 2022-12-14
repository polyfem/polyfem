#pragma once

#include "Parameter.hpp"

#include <ipc/collisions/collision_constraint.hpp>
#include <ipc/broad_phase/broad_phase.hpp>
#include "constraints/ShapeConstraints.hpp"

namespace polyfem
{
	class GraphParameter : public Parameter
	{
	public:
		GraphParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override { return initial_guess_; }

		Eigen::MatrixXd map(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const override;

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const override;

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		bool pre_solve(const Eigen::VectorXd &newX) override;

    private:

		bool generate_graph_mesh(const Eigen::VectorXd &x);

		double max_change_;
        std::string isosurface_inflator_prefix_; // binary to generate mesh based on graph and params
        Eigen::VectorXd initial_guess_; // initial shape parameters

		std::string out_velocity_path_;
		std::string out_msh_path_;
        Eigen::MatrixXd shape_velocity_; // chain rule from grad of vertices to grad of shape parameters

        Eigen::MatrixXd bounds_; // bound on shape parameters

		double unit_size_;
		std::vector<int> full_to_periodic_;
	
		void compute_pattern_period();
	};
} // namespace polyfem