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

// map BroadPhaseMethod values to JSON as strings
namespace ipc
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "sweep_and_tiniest_queue_gpu"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}})
} // namespace ipc

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
				solver_args["nonlinear"]);
		}
		else if (name == "newton" || name == "Newton")
		{
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				solver_args["nonlinear"], solver_args["linear"]);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(solver_args["nonlinear"]);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}
}

void State::solve_homogenized_field(const Eigen::MatrixXd &disp_grad, Eigen::MatrixXd &sol_, bool for_bistable)
{
    const int dim = mesh->dimension();
    const int ndof = n_bases * dim;

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
            collision_mesh, boundary_nodes_pos,
            args["contact"]["dhat"],
            avg_mass,
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
                boundary_nodes_pos,
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
        formulation(),
        boundary_nodes,
        local_boundary,
        n_boundary_samples(),
        *solve_data.rhs_assembler, *this, 0, forms);

    if (for_bistable)
    {
        homo_problem->set_only_symmetric();
    }

    if (args["optimization"]["enabled"])
    {
        solve_data.elastic_form = elastic_form;
        solve_data.contact_form = contact_form;
        solve_data.friction_form = friction_form;
        solve_data.nl_problem = homo_problem;
    }

    std::shared_ptr<cppoptlib::NonlinearSolver<NLHomoProblem>> nl_solver = make_nl_homo_solver<NLHomoProblem>(args["solver"]);

    Eigen::VectorXd tmp_sol;
    if (sol_.rows() != ndof || sol_.cols() != 1)
        sol_.setZero(ndof, 1);
    tmp_sol = homo_problem->full_to_reduced(sol_, Eigen::MatrixXd::Zero(dim, dim));
    Eigen::VectorXd tail = homo_problem->macro_full_to_reduced(utils::flatten(disp_grad));
    tmp_sol.tail(tail.size()) = tail;
    // export_data(homo_problem->reduced_to_full(tmp_sol), Eigen::MatrixXd());
    if (for_bistable)
    {
        homo_problem->set_fixed_entry({1, 2, 3});

        homo_problem->init(homo_problem->reduced_to_full(tmp_sol));
        nl_solver->minimize(*homo_problem, tmp_sol);

        homo_problem->set_fixed_entry({});

        // homo_problem->init(homo_problem->reduced_to_full(tmp_sol));
        nl_solver->minimize(*homo_problem, tmp_sol);
    }
    else
    {
        homo_problem->set_fixed_entry({0, 1, 2, 3});

        homo_problem->init(homo_problem->reduced_to_full(tmp_sol));
        nl_solver->minimize(*homo_problem, tmp_sol);
    }

    sol_ = homo_problem->reduced_to_full(tmp_sol);

    if (args["optimization"]["enabled"])
        cache_transient_adjoint_quantities(0, sol_, homo_problem->reduced_to_disp_grad(tmp_sol));

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
            collision_mesh, boundary_nodes_pos,
            args["contact"]["dhat"],
            avg_mass,
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
                boundary_nodes_pos,
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
        formulation(),
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