#include <polyfem/State.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/GradientDescentSolver.hpp>

#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>
#include <unsupported/Eigen/SparseExtra>

#include <polyfem/io/Evaluator.hpp>

#include <ipc/ipc.hpp>

namespace polyfem {

using namespace assembler;
using namespace mesh;
using namespace solver;
using namespace utils;
using namespace quadrature;

namespace
{
	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_homo_solver(const json &solver_args)
	{
		const std::string name = solver_args["nonlinear"]["solver"];
		if (name == "GradientDescent" || name == "gradientdescent" || name == "gradient")
		{
			return std::make_shared<cppoptlib::GradientDescentSolver<ProblemType>>(
				solver_args["nonlinear"], 0);
		}
		else if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				solver_args["nonlinear"], solver_args["linear"], 0);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(solver_args["nonlinear"], 0);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}
}

void State::solve_homogenized_field(const Eigen::MatrixXd &disp_grad, Eigen::MatrixXd &sol_, const std::vector<int> &fixed_entry, bool for_bistable)
{
    const int dim = mesh->dimension();
    const int ndof = n_bases * dim;

    Eigen::MatrixXd pressure;
    init_solve(sol_, pressure);

    solver::SolveData solve_data_tmp = solve_data;
    const std::vector<std::shared_ptr<Form>> forms = solve_data_tmp.init_forms(
        // General
        mesh->dimension(), 0,
        // Elastic form
        n_bases, bases, geom_bases(), assembler, ass_vals_cache, formulation(),
        // Body form
        n_pressure_bases, boundary_nodes, local_boundary, local_neumann_boundary,
        n_boundary_samples(), rhs, sol_,
        // Inertia form
        args["solver"]["ignore_inertia"], mass,
        // Lagged regularization form
        args["solver"]["advanced"]["lagged_regularization_weight"],
        args["solver"]["advanced"]["lagged_regularization_iterations"],
        // Augmented lagrangian form
        obstacle,
        // Contact form
        args["contact"]["enabled"], collision_mesh, args["contact"]["dhat"],
        avg_mass, args["contact"]["use_convergent_formulation"],
        args["solver"]["contact"]["barrier_stiffness"],
        args["solver"]["contact"]["CCD"]["broad_phase"],
        args["solver"]["contact"]["CCD"]["tolerance"],
        args["solver"]["contact"]["CCD"]["max_iterations"],
        // Friction form
        args["contact"]["friction_coefficient"],
        args["contact"]["epsv"],
        args["solver"]["contact"]["friction_iterations"],
        // Rayleigh damping form
        args["solver"]["rayleigh_damping"]);

    std::vector<int> boundary_nodes_tmp = boundary_nodes;
    full_to_periodic(boundary_nodes_tmp);

    std::shared_ptr<NLHomoProblem> homo_problem = std::make_shared<NLHomoProblem>(
        ndof,
        boundary_nodes_tmp,
        local_boundary,
        n_boundary_samples(),
        *solve_data_tmp.rhs_assembler, *this, 0, forms);
    solve_data_tmp.nl_problem = homo_problem;

    if (args["optimization"]["enabled"])
    {
        solve_data = solve_data_tmp;
    }

    homo_problem->set_fixed_entry(fixed_entry);
    {
        bool flag = false;
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (std::find(fixed_entry.begin(), fixed_entry.end(), i + j * dim) == fixed_entry.end() &&
                    std::find(fixed_entry.begin(), fixed_entry.end(), j + i * dim) == fixed_entry.end())
                {
                    logger().info("Strain entry [{},{}] and [{},{}] are not fixed, solve for symmetric strain...", i, j, j, i);
                    flag = true;
                    break;
                }
            }
        }
        if (flag)
        {
            if ((disp_grad - disp_grad.transpose()).norm() > 1e-10)
                log_and_throw_error("Macro strain is not symmetric!");
            homo_problem->set_only_symmetric();
        }
    }

    Eigen::VectorXd tmp_sol = homo_problem->full_to_reduced(sol_, Eigen::MatrixXd::Zero(dim, dim));
    Eigen::VectorXd tail = homo_problem->macro_full_to_reduced(utils::flatten(disp_grad));
    tmp_sol.tail(tail.size()) = tail;

    homo_problem->init(homo_problem->reduced_to_full(tmp_sol));
    std::shared_ptr<cppoptlib::NonlinearSolver<NLHomoProblem>> nl_solver = make_nl_homo_solver<NLHomoProblem>(args["solver"]);
    nl_solver->minimize(*homo_problem, tmp_sol);
    
    if (for_bistable)
    {
        homo_problem->set_fixed_entry({});
        nl_solver->minimize(*homo_problem, tmp_sol);
    }

    sol_ = homo_problem->reduced_to_full(tmp_sol);

    Eigen::MatrixXd disp_grad_out = homo_problem->reduced_to_disp_grad(tmp_sol);

    logger().info("displacement grad {}", utils::flatten(disp_grad_out).transpose());

    if (args["optimization"]["enabled"])
        cache_transient_adjoint_quantities(0, sol_, disp_grad_out);

    // static int index = 0;
    // StiffnessMatrix H;
    // homo_problem->hessian(tmp_sol, H);
    // Eigen::saveMarket(H, "H" + std::to_string(index) + ".mat");
    // index++;
}

void State::solve_homogenized_field_incremental(const Eigen::MatrixXd &macro_field2, Eigen::MatrixXd &macro_field1, Eigen::MatrixXd &sol_)
{
    const int dim = mesh->dimension();
    const int ndof = n_bases * dim;

    if (sol_.rows() != ndof || sol_.cols() != 1)
        sol_.setZero(ndof, 1);

    std::vector<std::shared_ptr<Form>> forms;

    std::shared_ptr<ElasticForm> elastic_form = std::make_shared<ElasticForm>(
        n_bases, bases, geom_bases(),
        assembler, ass_vals_cache,
        formulation(),
        problem->is_time_dependent() ? args["time"]["dt"].get<double>() : 0.0,
        mesh->is_volume());
    forms.push_back(elastic_form);

    std::shared_ptr<ContactForm> contact_form = nullptr;
    std::shared_ptr<FrictionForm> friction_form = nullptr;
    if (args["contact"]["enabled"])
    {

        const bool use_adaptive_barrier_stiffness = !args["solver"]["contact"]["barrier_stiffness"].is_number();

        contact_form = std::make_shared<ContactForm>(
            collision_mesh,
            args["contact"]["dhat"],
            avg_mass,
            args["contact"]["use_convergent_formulation"],
            use_adaptive_barrier_stiffness,
            /*is_time_dependent=*/solve_data.time_integrator != nullptr,
            args["solver"]["contact"]["CCD"]["broad_phase"],
            args["solver"]["contact"]["CCD"]["tolerance"],
            args["solver"]["contact"]["CCD"]["max_iterations"]);

        if (use_adaptive_barrier_stiffness)
        {
            contact_form->set_weight(1);
            logger().debug("Using adaptive barrier stiffness");
        }
        else
        {
            contact_form->set_weight(args["solver"]["contact"]["barrier_stiffness"]);
            logger().debug("Using fixed barrier stiffness of {}", contact_form->barrier_stiffness());
        }

        forms.push_back(contact_form);

        // ----------------------------------------------------------------

        if (args["contact"]["friction_coefficient"].get<double>() != 0)
        {
            friction_form = std::make_shared<FrictionForm>(
                collision_mesh,
                args["contact"]["epsv"],
                args["contact"]["friction_coefficient"],
                args["contact"]["dhat"],
                args["solver"]["contact"]["CCD"]["broad_phase"],
                args.value("/time/dt"_json_pointer, 1.0), // dt=1.0 if static
                *contact_form,
                args["solver"]["contact"]["friction_iterations"]);
            forms.push_back(friction_form);
        }
    }

    std::shared_ptr<NLHomoProblem> homo_problem = std::make_shared<NLHomoProblem>(
        ndof,
        boundary_nodes,
        local_boundary,
        n_boundary_samples(),
        *solve_data.rhs_assembler, *this, 0, forms);

    std::shared_ptr<cppoptlib::NonlinearSolver<NLHomoProblem>> nl_solver = make_nl_homo_solver<NLHomoProblem>(args["solver"]);

    Eigen::VectorXd tmp_sol = homo_problem->full_to_reduced(sol_, Eigen::MatrixXd::Zero(dim, dim));
    tmp_sol.tail(macro_field1.size()) = utils::flatten(macro_field1);
    // homo_problem->set_disp_offset(macro_field1);
    Eigen::MatrixXd cur_disp = homo_problem->reduced_to_full(tmp_sol);
    const Eigen::MatrixXd displaced = collision_mesh.displace_vertices(
				utils::unflatten(cur_disp, mesh->dimension()));
    if (!std::isfinite(homo_problem->value(tmp_sol))
        || ipc::has_intersections(collision_mesh, displaced))
    {
        args["output"]["paraview"]["file_name"] = "nan.vtu";
        export_data(cur_disp, Eigen::MatrixXd());
        log_and_throw_error("invalid last solution!");
    }

    Eigen::MatrixXd last_disp = cur_disp;
    int ind = 0;
    while (true)
    {
        double coeff = 1;
        Eigen::VectorXd tmp_macro_field = macro_field2;
        // homo_problem->set_disp_offset(tmp_macro_field);
        tmp_sol.tail(tmp_macro_field.size()) = utils::flatten(tmp_macro_field);
        cur_disp = homo_problem->reduced_to_full(tmp_sol);

        while (!std::isfinite(homo_problem->value(tmp_sol))
            || !homo_problem->is_step_valid(last_disp, cur_disp)
            || !homo_problem->is_step_collision_free(last_disp, cur_disp))
        {
            coeff /= 2;
            tmp_macro_field = coeff * macro_field2 + (1 - coeff) * macro_field1;
            tmp_sol.tail(tmp_macro_field.size()) = utils::flatten(tmp_macro_field);
            // homo_problem->set_disp_offset(tmp_macro_field);
            cur_disp = homo_problem->reduced_to_full(tmp_sol);

            logger().info("NAN detected, reduce step size to {}", coeff);

            if (coeff < 1e-16)
                log_and_throw_error("Failed to find a valid step!");
        }

        homo_problem->init(homo_problem->reduced_to_full(tmp_sol));
        nl_solver->minimize(*homo_problem, tmp_sol);
        last_disp = homo_problem->reduced_to_full(tmp_sol);

        out_geom.save_vtu(
            "debug_" + std::to_string(ind) + ".vtu",
            *this,
            last_disp,
            Eigen::MatrixXd(),
            1.0, 1.0,
            io::OutGeometryData::ExportOptions(args, mesh->is_linear(), problem->is_scalar(), solve_export_to_file),
            is_contact_enabled(),
            solution_frames);
        ind++;

        macro_field1 = tmp_macro_field;

        if (coeff == 1)
            break;
    }
    
    sol_ = homo_problem->reduced_to_full(tmp_sol);

    static int index = 0;
    StiffnessMatrix H;
    homo_problem->hessian(tmp_sol, H);
    Eigen::saveMarket(H, "H" + std::to_string(index) + ".mat");
    index++;
}

}