#include "TopologyOptimizationParameter.hpp"

namespace polyfem
{
    TopologyOptimizationParameter::TopologyOptimizationParameter(std::vector<std::shared_ptr<State>> states_ptr): Parameter(states_ptr)
    {
        parameter_name_ = "topology";

        full_dim_ = get_state().bases.size();
        optimization_dim_ = full_dim_;

        json opt_params = get_state().args["optimization"];
        bool found = false;
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "topology")
			{
				topo_params = param;
				if (param["bound"].get<std::vector<double>>().size() == 2)
				{
					min_density = param["bound"][0];
					max_density = param["bound"][1];
				}
                found = true;
				break;
			}
		}
        assert(found);

		for (const auto &param : opt_params["constraints"])
		{
			if (param["type"] == "mass")
			{
				has_mass_constraint = true;
				min_mass = param["bound"][0];
				max_mass = param["bound"][1];
                break;
			}
		}

        build_filter(topo_params["filter"]);
    }

    Eigen::MatrixXd TopologyOptimizationParameter::map(const Eigen::VectorXd &x) const
    {
        return apply_filter(x);
    }

    Eigen::VectorXd TopologyOptimizationParameter::map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const
    {
        return apply_filter_to_grad(x, full_grad);
    }

    Eigen::VectorXd TopologyOptimizationParameter::get_lower_bound(const Eigen::VectorXd &x) const
    {
        Eigen::VectorXd min(x.size());
        min.setConstant(min_density);
        for (int i = 0; i < min.size(); i++)
        {
            if (x(i) - min(i) > max_change)
                min(i) = x(i) - max_change;
        }
        return min;
    }

    Eigen::VectorXd TopologyOptimizationParameter::get_upper_bound(const Eigen::VectorXd &x) const
    {
        Eigen::VectorXd max(x.size());
        max.setConstant(max_density);
        for (int i = 0; i < max.size(); i++)
        {
            if (max(i) - x(i) > max_change)
                max(i) = x(i) + max_change;
        }
        return max;
    }

    bool TopologyOptimizationParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
    {
        const auto &cur_density = get_state().assembler.lame_params().density_mat_;

        if (cur_density.minCoeff() < min_density || cur_density.maxCoeff() > max_density)
            return false;
        
        return true;
    }

    Eigen::VectorXd TopologyOptimizationParameter::force_inequality_constraint(const Eigen::VectorXd &x0, const Eigen::VectorXd &dx)
    {
        return x0 + dx;
    }

    int TopologyOptimizationParameter::n_inequality_constraints()
    {
        // TODO: total mass constraint
        return 0;
    }

    double TopologyOptimizationParameter::inequality_constraint_val(const Eigen::VectorXd &x, const int index)
    {
        return -1;
    }

    Eigen::VectorXd TopologyOptimizationParameter::inequality_constraint_grad(const Eigen::VectorXd &x, const int index)
    {
        return Eigen::VectorXd::Zero(x.size());
    }

    bool TopologyOptimizationParameter::pre_solve(const Eigen::VectorXd &newX)
    {
        for (const auto &state : states_ptr_)
		    state->assembler.update_lame_params_density(apply_filter(newX));
		return true;
    }

    void TopologyOptimizationParameter::build_filter(const json &filter_args)
    {
        const double radius = filter_args["radius"];
        const State &state = get_state();
        std::vector<Eigen::Triplet<double>> tt_adjacency_list;

        Eigen::MatrixXd barycenters;
        if (state.mesh->is_volume())
            state.mesh->cell_barycenters(barycenters);
        else
            state.mesh->face_barycenters(barycenters);

        RowVectorNd min, max;
        state.mesh->bounding_box(min, max);
        // TODO: more efficient way
        for (int i = 0; i < state.mesh->n_elements(); i++)
        {
            auto center_i = barycenters.row(i);
            for (int j = 0; j <= i; j++)
            {
                auto center_j = barycenters.row(j);
                double dist = 0;
                dist = (center_i - center_j).norm();
                if (dist < radius)
                {
                    tt_adjacency_list.emplace_back(i, j, radius - dist);
                    if (i != j)
                        tt_adjacency_list.emplace_back(j, i, radius - dist);
                }
            }
        }
        tt_radius_adjacency.resize(state.mesh->n_elements(), state.mesh->n_elements());
        tt_radius_adjacency.setFromTriplets(tt_adjacency_list.begin(), tt_adjacency_list.end());

        tt_radius_adjacency_row_sum.setZero(tt_radius_adjacency.rows());
        for (int i = 0; i < tt_radius_adjacency.rows(); i++)
            tt_radius_adjacency_row_sum(i) = tt_radius_adjacency.row(i).sum();
    }

    Eigen::VectorXd TopologyOptimizationParameter::apply_filter(const Eigen::VectorXd &x) const
    {
        Eigen::VectorXd y = (tt_radius_adjacency * x).array() / tt_radius_adjacency_row_sum.array();
        return y;
    }

    Eigen::VectorXd TopologyOptimizationParameter::apply_filter_to_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const
    {
        Eigen::VectorXd grad_ = (tt_radius_adjacency * grad).array() / tt_radius_adjacency_row_sum.array();
        return grad_;
    }
}