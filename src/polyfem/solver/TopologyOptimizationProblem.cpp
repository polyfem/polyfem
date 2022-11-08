#include "TopologyOptimizationProblem.hpp"
#include <polyfem/solver/AdjointForm.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <filesystem>

namespace polyfem
{
    TopologyOptimizationProblem::TopologyOptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
    {
        optimization_name = "topology";
        state.args["output"]["paraview"]["options"]["topology"] = true;

		// volume constraint
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "topology")
			{
				top_params = param;
				if (param.contains("bound"))
				{
					min_density = param["bound"][0];
					max_density = param["bound"][1];
					state.min_solid_density = min_density;
				}
				break;
			}
		}

		// target functional
		for (const auto &param : opt_params["functionals"])
		{
			if (param["type"] == "homogenized_permeability" || param["type"] == "homogenized_stiffness")
			{
				if (param.contains("weight"))
					target_weight = param["weight"];
				break;
			}
		}

		// mass constraint
		has_mass_constraint = false;
		for (const auto &param : opt_params["constraints"])
		{
			if (param["type"] == "mass")
			{
				has_mass_constraint = true;
				min_mass = param["bound"][0];
				max_mass = param["bound"][1];
			}
		}

		// density filter
		has_filter = false;
		if (top_params.contains("filter"))
		{
			has_filter = true;
			const double radius = top_params["filter"]["radius"];
			std::vector<Eigen::Triplet<double>> tt_adjacency_list;
			Eigen::MatrixXd barycenters;
			if (state.mesh->is_volume())
				state.mesh->cell_barycenters(barycenters);
			else
				state.mesh->face_barycenters(barycenters);

			RowVectorNd min, max;
			state.mesh->bounding_box(min, max);
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
	}

	double TopologyOptimizationProblem::value(const TVector &x)
	{
		if (std::isnan(cur_val))
		{
			double target_val = target_value(x);
			logger().debug("target = {}", target_val);
			cur_val = target_val;
		}
		return cur_val;
	}

	void TopologyOptimizationProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		gradv = apply_filter_to_grad(x, j->gradient(state, "topology")) * target_weight;
	}

	void TopologyOptimizationProblem::gradient(const TVector &x, TVector &gradv)
	{
		if (cur_grad.size() == 0)
		{
			Eigen::VectorXd grad_target;
			target_gradient(x, grad_target);
			logger().debug("‖∇ target‖ = {}", grad_target.norm());
			cur_grad = grad_target;
		}

		gradv = cur_grad;
	}

    bool TopologyOptimizationProblem::is_step_valid(const TVector &x0, const TVector &x1)
    {
        const auto &cur_density = state.assembler.lame_params().density_mat_;

        if (cur_density.minCoeff() < min_density || cur_density.maxCoeff() > max_density)
            return false;
        
        return true;
    }

	cppoptlib::Problem<double>::TVector TopologyOptimizationProblem::force_inequality_constraint(const TVector &x0, const TVector &dx)
	{
		TVector x_new = x0 + dx;

		for (int i = 0; i < x_new.size(); i++)
		{
			if (x_new(i) < min_density)
				x_new(i) = min_density;
			else if (x_new(i) > max_density)
				x_new(i) = max_density;
		}

		return x_new;
	}

	bool TopologyOptimizationProblem::solution_changed_pre(const TVector &newX)
	{
		state.assembler.update_lame_params_density(apply_filter(newX));
		return true;
	}

	void TopologyOptimizationProblem::line_search_end()
	{
	}

	cppoptlib::Problem<double>::TVector TopologyOptimizationProblem::apply_filter(const TVector &x)
	{
		if (has_filter)
		{
			TVector y = (tt_radius_adjacency * x).array() / tt_radius_adjacency_row_sum.array();
			return y;
		}
		return x;
	}

	cppoptlib::Problem<double>::TVector TopologyOptimizationProblem::apply_filter_to_grad(const TVector &x, const TVector &grad)
	{
		if (has_filter)
		{
			TVector grad_ = (tt_radius_adjacency * grad).array() / tt_radius_adjacency_row_sum.array();
			return grad_;
		}
		return grad;
	}

	int TopologyOptimizationProblem::n_inequality_constraints()
	{
		if (!has_mass_constraint)
			return 0;
		
		int n_constraints = 1;
		if (min_mass > 0)
			n_constraints++;
		return n_constraints; // mass constraints
	}
	double TopologyOptimizationProblem::inequality_constraint_val(const cppoptlib::Problem<double>::TVector &x, const int index)
	{
		double val = 0;
		auto filtered_x = apply_filter(x);
		auto &gbases = state.geom_bases();
		for (int e = 0; e < state.bases.size(); e++)
		{
            assembler::ElementAssemblyValues vals;
            state.ass_vals_cache.compute(e, state.mesh->is_volume(), state.bases[e], gbases[e], vals);
			val += (vals.det.array() * vals.quadrature.weights.array()).sum() * filtered_x(e);
		}

		logger().debug("Current mass: {}, min {}, max {}", val, min_mass, max_mass);

		if (index == 0)
			val = val / max_mass - 1;
		else if (index == 1)
			val = 1 - val / min_mass;
		else
			assert(false);

		return val;
	}
	cppoptlib::Problem<double>::TVector TopologyOptimizationProblem::inequality_constraint_grad(const cppoptlib::Problem<double>::TVector &x, const int index)
	{
		IntegrableFunctional j;
		j.set_name("Mass");
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setOnes(u.rows(), 1);
			val *= params["density"].get<double>();
		});
		TVector grad;
		polyfem::solver::AdjointForm::gradient(state, j, "topology", grad, {}, polyfem::solver::AdjointForm::SpatialIntegralType::VOLUME, "final");
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
}