#include "TopologyOptimizationParameter.hpp"

namespace polyfem
{
    TopologyOptimizationParameter::TopologyOptimizationParameter(std::vector<std::shared_ptr<State>> states_ptr, const json &args): Parameter(states_ptr, args)
    {
        parameter_name_ = "topology";
        const auto &state = get_state();

        full_dim_ = state.bases.size() * 2;
        optimization_dim_ = args["periodic"] > 0 ? args["periodic"].get<int>() : state.bases.size();
        if (state.bases.size() % optimization_dim_)
            logger().error("Number of elements doesn't match with pattern periodicity!");

		assert(args["type"] == "topology");
        topo_params = args;
        if (args["bound"].get<std::vector<double>>().size() == 2)
        {
            min_density = args["bound"][0];
            max_density = args["bound"][1];
        }

        max_change = args["max_change"];

        if (args["initial"].is_number())
            initial_density_.setConstant(optimization_dim_, 1, args["initial"]);
        else
        {
            assert(args["initial"].is_array());
            assert(args["initial"].size() == 2);
            Eigen::VectorXd bound = args["initial"];
            std::srand(args["rand_seed"].get<int>());
            initial_density_.setZero(optimization_dim_);
            for (int i = 0; i < initial_density_.size(); i++)
            {
                initial_density_(i) = (std::rand() / (double)RAND_MAX) * (bound[1] - bound[0]) + bound[0];
            }
        }
        density_power_ = args["power"];
        lambda0 = state.assembler.lame_params().lambda_mat_(0);
        mu0 = state.assembler.lame_params().mu_mat_(0);

        if (args.contains("constraints") && args["constraints"].size() != 0)
        {
            for (const auto &param : args["constraints"])
            {
                if (param["type"] == "mass")
                {
                    has_mass_constraint = true;
                    min_mass = param["bound"][0];
                    max_mass = param["bound"][1];
                    break;
                }
            }
        }

        has_filter = !topo_params["filter"].is_null();
        if (has_filter)
            build_filter(topo_params["filter"]);

        assert(is_step_valid(initial_guess(), initial_guess()));
        pre_solve(initial_guess());
    }

    Eigen::VectorXd TopologyOptimizationParameter::initial_guess() const
    {
        return initial_density_;
    }

    Eigen::MatrixXd TopologyOptimizationParameter::map(const Eigen::VectorXd &x) const
    {
        assert(false);
        return apply_filter(x);
    }

    Eigen::VectorXd TopologyOptimizationParameter::map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const
    {
        const int n_elem = full_grad.size() / 2;
        const Eigen::VectorXd &dJ_dlambda = full_grad.head(n_elem);
        const Eigen::VectorXd &dJ_dmu = full_grad.tail(n_elem);
        const auto &state = get_state();
        const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
        const auto &cur_mus = state.assembler.lame_params().mu_mat_;
        const Eigen::VectorXd filtered_density = apply_filter(x);

        Eigen::VectorXd dJ_drho(n_elem);
        for (int e = 0; e < n_elem; e++)
        {
            const double E = convert_to_E(state.mesh->is_volume(), cur_lambdas(e), cur_mus(e));
            const double nu = convert_to_nu(state.mesh->is_volume(), cur_lambdas(e), cur_mus(e));
            const Eigen::Matrix2d jacobian = d_lambda_mu_d_E_nu(state.mesh->is_volume(), E, nu);
            
            const double dJ_dE = dJ_dlambda(e) * jacobian(0, 0) + dJ_dmu(e) * jacobian(1, 0);
            dJ_drho(e) = dJ_dE * density_power_ * E / filtered_density(e);
        }
        return apply_filter_to_grad(x, dJ_drho);
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
        if (x1.minCoeff() < min_density || x1.maxCoeff() > max_density)
            return false;
        
        return true;
    }

    Eigen::VectorXd TopologyOptimizationParameter::force_inequality_constraint(const Eigen::VectorXd &x0, const Eigen::VectorXd &dx)
    {
		Eigen::VectorXd x_new = x0 + dx;

		for (int i = 0; i < x_new.size(); i++)
		{
			if (x_new(i) < min_density)
				x_new(i) = min_density;
			else if (x_new(i) > max_density)
				x_new(i) = max_density;
		}

		return x_new;
    }

    int TopologyOptimizationParameter::n_inequality_constraints()
    {
		if (!has_mass_constraint)
			return 0;
		
		int n_constraints = 1;
		if (min_mass > 0)
			n_constraints++;
		return n_constraints; // mass constraints
    }

    double TopologyOptimizationParameter::inequality_constraint_val(const Eigen::VectorXd &x, const int index)
    {
		auto filtered_density = apply_filter(x);
        const auto &state = get_state();
		auto &gbases = state.geom_bases();

        double val = 0;
		for (int e = 0; e < state.bases.size(); e++)
		{
            assembler::ElementAssemblyValues vals;
            state.ass_vals_cache.compute(e, state.mesh->is_volume(), state.bases[e], gbases[e], vals);
			val += (vals.det.array() * vals.quadrature.weights.array()).sum() * filtered_density(e);
		}

		if (index == 0)
			val = val / max_mass - 1;
		else if (index == 1)
			val = 1 - val / min_mass;
		else
			assert(false);
        
        return val;
    }

    Eigen::VectorXd TopologyOptimizationParameter::inequality_constraint_grad(const Eigen::VectorXd &x, const int index)
    {
		auto filtered_density = apply_filter(x);
        const auto &state = get_state();
		auto &gbases = state.geom_bases();

        Eigen::VectorXd grad(state.bases.size());
		for (int e = 0; e < state.bases.size(); e++)
		{
            assembler::ElementAssemblyValues vals;
            state.ass_vals_cache.compute(e, state.mesh->is_volume(), state.bases[e], gbases[e], vals);
			grad(e) = (vals.det.array() * vals.quadrature.weights.array()).sum();
		}

        grad = apply_filter_to_grad(x, grad);

		if (index == 0)
		{
			grad /= max_mass;
		}
		else if (index == 1)
		{
			grad *= -1 / min_mass;
		}
		else
			assert(false);

		return grad;
    }

    bool TopologyOptimizationParameter::pre_solve(const Eigen::VectorXd &newX)
    {
        const Eigen::VectorXd filtered_density = apply_filter(newX);

        for (auto &state : states_ptr_)
        {
            auto cur_lambdas = state->assembler.lame_params().lambda_mat_;
            auto cur_mus = state->assembler.lame_params().mu_mat_;

            assert(cur_mus.size() == filtered_density.size());
            assert(cur_lambdas.size() == filtered_density.size());
            for (int e = 0; e < cur_mus.size(); e++)
            {
                cur_mus(e) = pow(filtered_density(e), density_power_) * mu0;
                cur_lambdas(e) = pow(filtered_density(e), density_power_) * lambda0;
            }
            state->assembler.update_lame_params(cur_lambdas, cur_mus);
        }

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
        for (int i = 0; i < optimization_dim_; i++)
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
        tt_radius_adjacency.resize(optimization_dim_, optimization_dim_);
        tt_radius_adjacency.setFromTriplets(tt_adjacency_list.begin(), tt_adjacency_list.end());

        tt_radius_adjacency_row_sum.setZero(tt_radius_adjacency.rows());
        for (int i = 0; i < tt_radius_adjacency.rows(); i++)
            tt_radius_adjacency_row_sum(i) = tt_radius_adjacency.row(i).sum();
    }

    Eigen::VectorXd TopologyOptimizationParameter::apply_filter(const Eigen::VectorXd &x) const
    {
        Eigen::VectorXd y;
        if (!has_filter)
            y = x;
        else
            y = (tt_radius_adjacency * x).array() / tt_radius_adjacency_row_sum.array();

        int n_tiles = get_state().bases.size() / optimization_dim_;
        Eigen::VectorXd y_full = y.replicate(n_tiles, 1);
        return y_full;
    }

    Eigen::VectorXd TopologyOptimizationParameter::apply_filter_to_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const
    {
        int n_tiles = get_state().bases.size() / optimization_dim_;
        Eigen::VectorXd grad_reduced;
        grad_reduced.setZero(optimization_dim_);
        for (int i = 0; i < n_tiles; i++)
            grad_reduced += grad.segment(i * optimization_dim_, optimization_dim_);
        if (!has_filter)
            return grad_reduced;
        Eigen::VectorXd grad_ = (tt_radius_adjacency * grad_reduced).array() / tt_radius_adjacency_row_sum.array();
        return grad_;
    }
}