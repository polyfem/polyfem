#include <polyfem/State.hpp>
#include <polyfem/solver/HomogenizationNLProblem.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>

#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
// #include <polyfem/solver/forms/LeastSquareForm.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>
#include <unsupported/Eigen/SparseExtra>

namespace polyfem {

using namespace assembler;
using namespace mesh;
using namespace solver;
using namespace utils;
using namespace quadrature;

namespace
{
    class LocalThreadMatStorage
    {
    public:
        SpareMatrixCache cache;
        ElementAssemblyValues vals;

        LocalThreadMatStorage()
        {
        }

        LocalThreadMatStorage(const int buffer_size, const int rows, const int cols)
        {
            init(buffer_size, rows, cols);
        }

        LocalThreadMatStorage(const int buffer_size, const SpareMatrixCache &c)
        {
            init(buffer_size, c);
        }

        void init(const int buffer_size, const int rows, const int cols)
        {
            // assert(rows == cols);
            cache.reserve(buffer_size);
            cache.init(rows, cols);
        }

        void init(const int buffer_size, const SpareMatrixCache &c)
        {
            cache.reserve(buffer_size);
            cache.init(c);
        }
    };

    class LocalThreadVecStorage
    {
    public:
        Eigen::MatrixXd vec;
        ElementAssemblyValues vals;

        LocalThreadVecStorage(const int size)
        {
            vec.resize(size, 1);
            vec.setZero();
        }
    };

    class LocalThreadScalarStorage
    {
    public:
        double val;
        ElementAssemblyValues vals;

        LocalThreadScalarStorage()
        {
            val = 0;
        }
    };
    
	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_homo_solver(const json &solver_args)
	{
		const std::string name = solver_args["nonlinear"]["solver"];
		if (name == "newton" || name == "Newton")
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

    Eigen::MatrixXd generate_linear_field(const State &state, const Eigen::MatrixXd &grad)
    {
        const int problem_dim = grad.rows();
        const int dim = state.mesh->dimension();
        assert(dim == grad.cols());

        Eigen::MatrixXd func(state.n_bases * problem_dim, 1);
        func.setZero();

        for (int i = 0; i < state.n_bases; i++)
        {
            func.block(i * problem_dim, 0, problem_dim, 1) = grad * state.mesh_nodes->node_position(i).transpose();
        }

        return func;
    }
}

void State::solve_homogenized_field(const Eigen::MatrixXd &disp_grad, const Eigen::MatrixXd &target, Eigen::MatrixXd &sol_, const std::string &hessian_path)
{
    if (formulation() != "NeoHookean" && formulation() != "LinearElasticity")
    {
        log_and_throw_error("Nonlinear homogenization only supports NeoHookean and linear elasticity!");
    }

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

    std::shared_ptr<HomogenizationNLProblem> homo_problem = std::make_shared<HomogenizationNLProblem>(
        ndof,
        formulation(),
        boundary_nodes,
        local_boundary,
        n_boundary_samples(),
        *solve_data.rhs_assembler, *this, 0, forms);
    
    Eigen::VectorXd macro_field = generate_linear_field(*this, disp_grad);
    homo_problem->set_macro_field(macro_field);

    std::shared_ptr<cppoptlib::NonlinearSolver<HomogenizationNLProblem>> nl_solver = make_nl_homo_solver<HomogenizationNLProblem>(args["solver"]);
    
    Eigen::VectorXd tmp_sol = homo_problem->full_to_reduced(sol_);
    homo_problem->init(tmp_sol);
    nl_solver->minimize(*homo_problem, tmp_sol);
    sol_ = homo_problem->reduced_to_full(tmp_sol);

    if (hessian_path != "")
    {
        StiffnessMatrix hessian;
        homo_problem->hessian(tmp_sol, hessian);
        Eigen::saveMarket(hessian, hessian_path);
    }
}

}