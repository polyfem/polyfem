#include <polyfem/State.hpp>
#include <polyfem/solver/NLHomogenizationProblem.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

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
}

Eigen::MatrixXd State::generate_linear_field(const Eigen::MatrixXd &grad)
{
    const int problem_dim = grad.rows();
    const int dim = mesh->dimension();
    assert(dim == grad.cols());

    Eigen::MatrixXd func(n_bases * problem_dim, 1);
    func.setZero();

    for (int i = 0; i < n_bases; i++)
    {
        func.block(i * problem_dim, 0, problem_dim, 1) = grad * mesh_nodes->node_position(i).transpose();
    }

    return func;
}

void State::solve_nonlinear_homogenization()
{
    assert(!assembler.is_linear(formulation())); // non-linear
    assert(!problem->is_scalar());               // tensor
    assert(!assembler.is_mixed(formulation()));

    if (formulation() != "NeoHookean")
    {
        logger().error("Nonlinear homogenization only supports NeoHookean!");
        return;
    }

    auto homo_problem = std::make_shared<NLHomogenizationProblem>(*this);
    
    const int dim = mesh->dimension();
    solver_info = json::array();
    sol.setZero(n_bases * dim, dim * dim);

    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            logger().info("Solve NeoHookean Homogenization index ({},{}) ...", i, j);
            Eigen::VectorXd tmp_sol;
            homo_problem->full_to_reduced(sol.col(i * dim + j), tmp_sol);

            Eigen::MatrixXd unit_grad;
            unit_grad.setZero(dim, dim);
            unit_grad(i, j) = nl_homogenization_scale;

            homo_problem->set_test_strain(unit_grad);

            std::shared_ptr<cppoptlib::NonlinearSolver<NLHomogenizationProblem>> nl_solver = make_nl_homo_solver<NLHomogenizationProblem>(args["solver"]);
            nl_solver->set_line_search(args["solver"]["nonlinear"]["line_search"]["method"]);
            homo_problem->init(tmp_sol);
            nl_solver->minimize(*homo_problem, tmp_sol);

            json nl_solver_info;
            nl_solver->get_info(nl_solver_info);
            solver_info.push_back(
                {{"type", "rc"},
                 {"info", nl_solver_info}});
            Eigen::VectorXd full;
            homo_problem->reduced_to_full(tmp_sol, full);
            sol.col(i * dim + j) = full;
        }
    }
}

void State::solve_linear_homogenization()
{
    if (stiffness.rows() <= 0)
    {
        logger().error("Assemble the stiffness matrix first!");
        return;
    }
    if (args["space"]["advanced"]["periodic_basis"])
    {
        logger().error("Homogenization doesn't support periodic basis!");
        return;
    }

    const int dim = mesh->dimension();
    const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
    if (formulation() == "LinearElasticity")
    {
        std::vector<std::pair<int, int>> unit_disp_ids;
        if (dim == 2)
            unit_disp_ids = {{0, 0}, {1, 1}, {0, 1}};
        else
            unit_disp_ids = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
        
        Eigen::MatrixXd unit_grad(dim, dim);
        rhs.setZero(stiffness.rows(), unit_disp_ids.size());
        Eigen::MatrixXd tmp_rhs;
        for (int i = 0; i < unit_disp_ids.size(); i++)
        {
            const auto &pair = unit_disp_ids[i];
            unit_grad.setZero();
            unit_grad(pair.first, pair.second) = 1;
            
            // assemble_homogenization_gradient(tmp_rhs, Eigen::MatrixXd::Zero(n_bases * problem_dim, 1), unit_grad);
			Eigen::MatrixXd test_field = generate_linear_field(unit_grad);
			
			assembler.assemble_energy_gradient(formulation(), mesh->is_volume(), n_bases, bases, geom_bases(), ass_vals_cache, test_field, tmp_rhs);

            rhs.col(i) = tmp_rhs;
        }
    }
    else
    {
        logger().error("Assembler not implemented for {}!", formulation());
        return;
    }

    // const std::string full_mat_path = args["output"]["data"]["full_mat"];
    // if (!full_mat_path.empty())
    // {
    //     Eigen::saveMarket(stiffness, full_mat_path);
    // }

    auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);
    StiffnessMatrix A = stiffness;
    const int full_size = A.rows();
    Eigen::VectorXd b;
    logger().info("{}...", solver->name());
    for (int b : boundary_nodes)
    {
        rhs.row(b).setZero();
    }
    int precond_num = problem_dim * n_bases;

    Eigen::VectorXd x;

    apply_lagrange_multipliers(A);
    rhs.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), rhs.cols()));

    if (args["boundary_conditions"]["periodic_boundary"])
    {
        precond_num = full_to_periodic(A);
        full_to_periodic(rhs);
    }

    sol.setZero(rhs.rows(), rhs.cols());
    
    StiffnessMatrix A_tmp = A;
    prefactorize(*solver, A_tmp, boundary_nodes, precond_num, args["output"]["data"]["stiffness_mat"]);
    for (int k = 0; k < rhs.cols(); k++)
    {
        b = rhs.col(k);
        dirichlet_solve_prefactorized(*solver, A, b, boundary_nodes, x);
        sol.col(k) = x;
    }
    // spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
    solver->getInfo(solver_info);

    const auto error = (A_tmp * sol - rhs).norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    sol.conservativeResize(sol.rows() - n_lagrange_multipliers(), sol.cols());
    rhs.conservativeResize(rhs.rows() - n_lagrange_multipliers(), rhs.cols());

    if (args["boundary_conditions"]["periodic_boundary"])
    {
        sol = periodic_to_full(full_size, sol);
        rhs = periodic_to_full(full_size, rhs);
    }

    if (assembler.is_mixed(formulation()))
    {
        sol_to_pressure();
    }
}

void State::solve_adjoint_homogenize_linear_elasticity(Eigen::MatrixXd &react_sol, Eigen::MatrixXd &adjoint_solution)
{
    const int dim = mesh->dimension();
    const auto &gbases = geom_bases();
    const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
    int precond_num = problem_dim * n_bases;

    std::vector<std::pair<int, int>> unit_disp_ids;
    if (dim == 2)
        unit_disp_ids = {{0, 0}, {1, 1}, {0, 1}};
    else
        unit_disp_ids = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

    const LameParameters &params = assembler.lame_params();
    
    std::vector<Eigen::MatrixXd> unit_strains(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));
    for (int id = 0; id < unit_disp_ids.size(); id++)
    {
        const auto &pair = unit_disp_ids[id];
        auto &unit_strain = unit_strains[id];

        Eigen::MatrixXd grad_unit(dim, dim);
        grad_unit.setZero();
        grad_unit(pair.first, pair.second) = 1;
        unit_strain = (grad_unit + grad_unit.transpose()) / 2;
    }

    StiffnessMatrix A = stiffness;
    const int full_size = A.rows();
    
    Eigen::MatrixXd adjoint_rhs;
    adjoint_rhs.setZero(stiffness.rows(), unit_disp_ids.size());
    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;
        
        for (int q = 0; q < quadrature.weights.size(); q++)
        {
            double lambda, mu;
            params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu, true);

            std::vector<Eigen::MatrixXd> react_strains(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));

            for (int id = 0; id < unit_disp_ids.size(); id++)
            {
                Eigen::MatrixXd grad_react(dim, dim);
                grad_react.setZero();
                for (const auto &v : vals.basis_values)
                    for (int d = 0; d < dim; d++)
                    {
                        double coeff = 0;
                        for (const auto &g : v.global)
                            coeff += react_sol(g.index * dim + d, id) * g.val;
                        grad_react.row(d) += v.grad_t_m.row(q) * coeff;
                    }
                
                react_strains[id] = (grad_react + grad_react.transpose()) / 2;
            }

            for (const auto &v : vals.basis_values)
            {
                for (int d = 0; d < dim; d++)
                {
                    Eigen::MatrixXd basis_strain, grad_basis;
                    basis_strain.setZero(dim, dim);
                    grad_basis.setZero(dim, dim);
                    grad_basis.row(d) = v.grad_t_m.row(q);
                    basis_strain = (grad_basis + grad_basis.transpose()) / 2; 
                
                    for (int id = 0; id < unit_disp_ids.size(); id++)
                    {
                        auto diff_strain = react_strains[id] - unit_strains[id];
                        
                        const double value = quadrature.weights(q) * vals.det(q) * (2 * mu * (diff_strain.array() * basis_strain.array()).sum() + lambda * diff_strain.trace() * basis_strain.trace());

                        for (auto g : v.global)
                            adjoint_rhs(g.index * dim + d, id) -= value * g.val;
                    }
                }
            }
        }
    }

    apply_lagrange_multipliers(A);
    adjoint_rhs.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), adjoint_rhs.cols()));

    if (args["boundary_conditions"]["periodic_boundary"] && !args["space"]["advanced"]["periodic_basis"])
    {
        precond_num = full_to_periodic(A);
        full_to_periodic(adjoint_rhs);
    }

    auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);
    {
        auto A_tmp = A;
        prefactorize(*solver, A_tmp, boundary_nodes, precond_num, args["output"]["data"]["stiffness_mat"]);
    }

    adjoint_solution.setZero(adjoint_rhs.rows(), adjoint_rhs.cols());
    for (int k = 0; k < adjoint_rhs.cols(); k++)
    {
        Eigen::VectorXd b = adjoint_rhs.col(k);
        Eigen::VectorXd x = adjoint_solution.col(k);
        dirichlet_solve_prefactorized(*solver, A, b, boundary_nodes, x);
        adjoint_solution.col(k) = x;
    }

    const double error = (A * adjoint_solution - adjoint_rhs).norm();
    if (std::isnan(error) || error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);
    
    adjoint_solution.conservativeResize(adjoint_solution.rows() - n_lagrange_multipliers(), adjoint_solution.cols());

    if (args["boundary_conditions"]["periodic_boundary"])
        adjoint_solution = periodic_to_full(full_size, adjoint_solution);
}

void State::compute_homogenized_tensor(Eigen::MatrixXd &C_H)
{
    if (stiffness.size() <= 0 && assembler.is_linear(formulation()))
    {
        logger().error("Assemble the matrix first!");
        return;
    }
    if (sol.size() <= 0)
    {
        logger().error("Solve the problem first!");
        return;
    }
    if (args["space"]["advanced"]["periodic_basis"])
    {
        logger().error("This homogenization doesn't support periodic basis!");
        return;
    }

    const int dim = mesh->dimension();
    const auto &gbases = geom_bases();
    RowVectorNd min, max;
    mesh->bounding_box(min, max);
    double volume = 1;
    for (int d = 0; d < min.size(); d++)
        volume *= (max(d) - min(d));
    if (formulation() == "LinearElasticity")
    {
        std::vector<std::pair<int, int>> unit_disp_ids;
        if (dim == 2)
            unit_disp_ids = {{0, 0}, {1, 1}, {0, 1}};
        else
            unit_disp_ids = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

        C_H.setZero(unit_disp_ids.size(), unit_disp_ids.size());

        Eigen::MatrixXd test_fields(sol.rows(), unit_disp_ids.size());
        for (int id = 0; id < unit_disp_ids.size(); id++)
        {
            const auto &pair = unit_disp_ids[id];

            Eigen::MatrixXd unit_grad(dim, dim);
            unit_grad.setZero();
            unit_grad(pair.first, pair.second) = 1;

            test_fields.col(id) = generate_linear_field(unit_grad);
        }

        Eigen::MatrixXd diff_fields = test_fields - sol;
        for (int i = 0; i < C_H.rows(); i++)
            for (int j = 0; j < C_H.cols(); j++)
                C_H(i, j) = diff_fields.col(i).transpose() * stiffness * diff_fields.col(j);
    }
    else if (formulation() == "NeoHookean")
    {
        C_H.setZero(dim*dim, dim*dim);

        const LameParameters &params = assembler.lame_params();

        for (int e = 0; e < bases.size(); e++)
        {
            ElementAssemblyValues vals;
            // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
            ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

            const Quadrature &quadrature = vals.quadrature;

            std::vector<Eigen::MatrixXd> sol_grads(dim*dim, Eigen::MatrixXd::Zero(dim, dim));
            for (int q = 0; q < quadrature.weights.size(); q++)
            {
                double lambda, mu;
                params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);

                for (int i = 0; i < dim; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        sol_grads[i * dim + j].setZero();
                        for (const auto &v : vals.basis_values)
                            for (const auto &g : v.global)
                                for (int d = 0; d < dim; d++)
                                sol_grads[i * dim + j].row(d) += g.val * sol(g.index * dim + d, i * dim + j) * v.grad_t_m.row(q);
                    }
                }

                for (int i = 0; i < dim; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        Eigen::MatrixXd def_grad = sol_grads[i * dim + j] + Eigen::MatrixXd::Identity(dim, dim);
                        def_grad(i, j) -= nl_homogenization_scale;
                        Eigen::MatrixXd FmT = def_grad.inverse().transpose();
                        Eigen::MatrixXd stress_ij = mu * (def_grad - FmT) + lambda * std::log(def_grad.determinant()) * FmT;

                        for (int k = 0; k < dim; k++)
                        {
                            for (int l = 0; l < dim; l++)
                            {
                                Eigen::MatrixXd grad_kl = sol_grads[k * dim + l];
                                grad_kl(k, l) -= nl_homogenization_scale;

                                C_H(i * dim + j, k * dim + l) += (stress_ij.array() * grad_kl.array()).sum() * vals.det(q) * quadrature.weights(q);
                            }
                        }
                    }
                }
            }
        }
    }
    else if (formulation() == "Stokes")
    {
        C_H.setZero(dim, dim);

        auto velocity_block = stiffness.topLeftCorner(n_bases * dim, n_bases * dim);
        for (int i = 0; i < dim; i++)
        {
            Eigen::VectorXd tmp = velocity_block * sol.block(0, i, n_bases * dim, 1);
            for (int j = 0; j < dim; j++)
                C_H(i, j) = (tmp.array() * sol.block(0, j, n_bases * dim, 1).array()).sum();
        }
    }
    C_H /= nl_homogenization_scale*nl_homogenization_scale*volume;
}

void State::homogenize_weighted_linear_elasticity(Eigen::MatrixXd &C_H)
{
    if (!mesh)
    {
        logger().error("Load the mesh first!");
        return;
    }
    if (n_bases <= 0)
    {
        logger().error("Build the bases first!");
        return;
    }
    if (formulation() != "LinearElasticity")
    {
        logger().error("Wrong formulation!");
        return;
    }
    if (!args["space"]["advanced"]["periodic_basis"])
    {
        logger().error("This homogenization only supports periodic basis!");
        return;
    }

    assemble_stiffness_mat();
    
    const int dim = mesh->dimension();
    const auto &gbases = geom_bases();
    
    std::vector<std::pair<int, int>> unit_disp_ids;
    if (dim == 2)
        unit_disp_ids = {{0, 0}, {1, 1}, {0, 1}};
    else
        unit_disp_ids = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

    Eigen::MatrixXd rhs; //, unit_disp;
    // unit_disp.setZero(stiffness.rows(), unit_disp_ids.size());
    rhs.setZero(stiffness.rows(), unit_disp_ids.size());

    const LameParameters &params = assembler.lame_params();
    
    std::vector<Eigen::MatrixXd> unit_strains(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));
    for (int id = 0; id < unit_disp_ids.size(); id++)
    {
        const auto &pair = unit_disp_ids[id];
        auto &unit_strain = unit_strains[id];

        Eigen::MatrixXd grad(dim, dim);
        grad.setZero();
        grad(pair.first, pair.second) = 1;
        unit_strain = (grad + grad.transpose()) / 2;
    }

    // for (int p = 0; p < nodes_position.rows(); p++)
    // {
    //     for (int k = 0; k < unit_disp_ids.size(); k++)
    //     {
    //         unit_disp(p * dim + unit_disp_ids[k].first, k) = nodes_position(p, unit_disp_ids[k].second);
    //     }
    // }

    // rhs = stiffness * unit_disp;

    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;

        for (int q = 0; q < quadrature.weights.size(); q++)
        {
            double lambda, mu;
            params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);

            for (const auto &v : vals.basis_values)
            {
                for (int d = 0; d < dim; d++)
                {
                    Eigen::MatrixXd basis_strain, grad;
                    basis_strain.setZero(dim, dim);
                    grad.setZero(dim, dim);
                    grad.row(d) = v.grad_t_m.row(q);
                    basis_strain = (grad + grad.transpose()) / 2; 

                    for (int k = 0; k < unit_disp_ids.size(); k++)
                    {
                        const auto &unit_strain = unit_strains[k];

                        const double value = quadrature.weights(q) * vals.det(q) * (2 * mu * (unit_strain.array() * basis_strain.array()).sum() + lambda * unit_strain.trace() * basis_strain.trace());

                        for (auto g : v.global)
                            rhs(g.index * dim + d, k) += value * g.val;
                    }
                }
            }
        }
    }

    // solve elastic problem
    StiffnessMatrix A = stiffness;
    const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
    int precond_num = problem_dim * n_bases;

    if (boundary_nodes.size() > 0)
    {
        logger().error("Homogenization with Dirichlet BC not implemented!");
        return;
    }

    apply_lagrange_multipliers(A);
    rhs.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), rhs.cols()));

    Eigen::MatrixXd w;
    w.setZero(rhs.rows(), rhs.cols());
    auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);
    auto A_tmp = A;
    {
        prefactorize(*solver, A_tmp, boundary_nodes, precond_num, args["output"]["data"]["stiffness_mat"]);
    }
    for (int k = 0; k < rhs.cols(); k++)
    {
        // auto A_tmp = A;
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        dirichlet_solve_prefactorized(*solver, A, b, boundary_nodes, x);
        // polysolve::dirichlet_solve(*solver, A_tmp, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
        // solver->getInfo(solver_info);
        w.col(k) = x;
    }

    const auto error = (A_tmp * w - rhs).norm();
    if (std::isnan(error) || error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);
    
    w.conservativeResize(n_bases * dim, w.cols());

    // auto diff = unit_disp - w;
    for (int id = 0; id < unit_disp_ids.size(); id++)
    {       
        sol = w.col(id);
        save_vtu("homo_" + std::to_string(unit_disp_ids[id].first) + std::to_string(unit_disp_ids[id].second) + ".vtu", 1.);
    }

    // compute homogenized stiffness
    C_H.setZero(unit_disp_ids.size(), unit_disp_ids.size());
    // for (int i = 0; i < C_H.rows(); i++)
    //     for (int j = 0; j < C_H.cols(); j++)
    //         C_H(i, j) = diff.col(i).transpose() * stiffness * diff.col(j);
    
    RowVectorNd min, max;
    mesh->bounding_box(min, max);
    double volume = 1;
    for (int d = 0; d < min.size(); d++)
        volume *= (max(d) - min(d));
    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;

        for (int q = 0; q < quadrature.weights.size(); q++)
        {
            double lambda, mu;
            params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);

            std::vector<Eigen::MatrixXd> react_strains(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));

            for (int id = 0; id < unit_disp_ids.size(); id++)
            {
                Eigen::MatrixXd grad(dim, dim);
                grad.setZero();
                for (const auto &v : vals.basis_values)
                    for (int d = 0; d < dim; d++)
                    {
                        double coeff = 0;
                        for (const auto &g : v.global)
                            coeff += w(g.index * dim + d, id) * g.val;
                        grad.row(d) += v.grad_t_m.row(q) * coeff;
                    }
                
                react_strains[id] = (grad + grad.transpose()) / 2;
            }

            for (int row_id = 0; row_id < unit_disp_ids.size(); row_id++)
            {
                Eigen::MatrixXd strain_diff_ij = unit_strains[row_id] - react_strains[row_id];

                for (int col_id = 0; col_id < unit_disp_ids.size(); col_id++)
                {
                    Eigen::MatrixXd strain_diff_kl = unit_strains[col_id] - react_strains[col_id];

                    const double value = 2 * mu * (strain_diff_ij.array() * strain_diff_kl.array()).sum() + lambda * strain_diff_ij.trace() * strain_diff_kl.trace();
                    C_H(row_id, col_id) += vals.quadrature.weights(q) * vals.det(q) * value;
                }
            }
        }
    }

    C_H /= volume;
}

void State::homogenize_linear_elasticity_shape_grad(Eigen::MatrixXd &C_H, Eigen::MatrixXd &grad)
{
    if (!mesh)
    {
        logger().error("Load the mesh first!");
        return;
    }
    if (n_bases <= 0)
    {
        logger().error("Build the bases first!");
        return;
    }
    if (formulation() != "LinearElasticity")
    {
        logger().error("Wrong formulation!");
        return;
    }

    const int dim = mesh->dimension();

    assemble_stiffness_mat();
    solve_homogenization();
    compute_homogenized_tensor(C_H);
    std::cout << "\n" << C_H << "\n";

    std::vector<std::pair<int, int>> unit_disp_ids;
    if (dim == 2)
        unit_disp_ids = {{0, 0}, {1, 1}, {0, 1}};
    else
        unit_disp_ids = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

    const LameParameters &params = assembler.lame_params();
    
    std::vector<Eigen::MatrixXd> unit_strains(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));
    std::vector<Eigen::MatrixXd> unit_grads(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));
    for (int id = 0; id < unit_disp_ids.size(); id++)
    {
        unit_grads[id].setZero(dim, dim);
        unit_grads[id](unit_disp_ids[id].first, unit_disp_ids[id].second) = 1;
        unit_strains[id] = (unit_grads[id] + unit_grads[id].transpose()) / 2;
    }

    Eigen::MatrixXd adjoint;
    solve_adjoint_homogenize_linear_elasticity(sol, adjoint);

    const auto &gbases = geom_bases();

    grad.setZero(dim * n_geom_bases, C_H.size());

    Eigen::VectorXd term;
    for (int a = 0; a < unit_disp_ids.size(); a++)
    {
        compute_shape_derivative_elasticity_term(sol.col(a), adjoint.col(a), term); // ignored the rhs contribution

        for (int b = 0; b <= a; b++)
        {
            grad.col(a * unit_disp_ids.size() + b) += term * (a == b ? 2 : 1);
        }
    }

    for (int e = 0; e < bases.size(); e++)
    {
        assembler::ElementAssemblyValues gvals, vals;
        gvals.compute(e, mesh->is_volume(), gbases[e], gbases[e]);

        const quadrature::Quadrature &quadrature = gvals.quadrature;
        const Eigen::VectorXd da = gvals.det.array() * quadrature.weights.array();

        vals.compute(e, mesh->is_volume(), quadrature.points, bases[e], gbases[e]);

        for (int q = 0; q < da.size(); ++q)
        {
            std::vector<Eigen::MatrixXd> diff_grads(unit_disp_ids.size()), adjoint_grads(unit_disp_ids.size());
            std::vector<Eigen::MatrixXd> diff_strains(unit_disp_ids.size()), adjoint_strains(unit_disp_ids.size());
            for (int a = 0; a < unit_disp_ids.size(); a++)
            {
                Eigen::MatrixXd grad_sol_a(dim, dim);
                grad_sol_a.setZero();
                for (const auto &v : vals.basis_values)
                    for (int d = 0; d < dim; d++)
                    {
                        double coeff = 0;
                        for (const auto &g : v.global)
                            coeff += sol(g.index * dim + d, a) * g.val;

                        grad_sol_a.row(d) += v.grad_t_m.row(q) * coeff;
                    }

                diff_grads[a] = grad_sol_a - unit_grads[a];
                diff_strains[a] = (diff_grads[a] + diff_grads[a].transpose()) / 2;

                adjoint_grads[a].setZero(dim, dim);
                for (const auto &v : vals.basis_values)
                    for (int d = 0; d < dim; d++)
                    {
                        double coeff = 0;
                        for (const auto &g : v.global)
                            coeff += adjoint(g.index * dim + d, a) * g.val;

                        adjoint_grads[a].row(d) += v.grad_t_m.row(q) * coeff;
                    }
                
                adjoint_strains[a] = (adjoint_grads[a] + adjoint_grads[a].transpose()) / 2;
            }

            double lambda, mu;
            params.lambda_mu(quadrature.points.row(q), gvals.val.row(q), e, lambda, mu);

            auto weak_form = [lambda, mu](const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
            {
                return 2 * mu * (A.array() * B.array()).sum() + lambda * A.trace() * B.trace();
            };

            for (auto &v : gvals.basis_values)
            {
                for (int d = 0; d < dim; d++)
                {
                    Eigen::MatrixXd grad_v_i;
                    grad_v_i.setZero(mesh->dimension(), mesh->dimension());
                    grad_v_i.row(d) = v.grad_t_m.row(q);

                    const double velocity_div = grad_v_i.trace();

                    for (int a = 0; a < unit_disp_ids.size(); a++)
                    {
                        for (int b = 0; b <= a; b++)
                        {
                            // d_q J
                            const double val1 = -weak_form(diff_strains[a], diff_grads[b] * grad_v_i);
                            const double val2 = -weak_form(diff_grads[a] * grad_v_i, diff_strains[b]);
                            const double val3 = weak_form(diff_strains[a], diff_strains[b]) * velocity_div;

                            // p^T d_q rhs
                            const double val4 = weak_form(unit_strains[a], adjoint_grads[a] * grad_v_i) - weak_form(unit_strains[a], adjoint_grads[a]) * velocity_div;
                            
                            grad(v.global[0].index * dim + d, a * unit_disp_ids.size() + b) += (val1 + val2 + val3 + val4 * (a == b ? 2 : 1)) * da(q);
                        }
                    }
                }
            }
        }
    }

    for (int a = 0; a < unit_disp_ids.size(); a++)
    {
        for (int b = a + 1; b < unit_disp_ids.size(); b++)
        {
            grad.col(a * unit_disp_ids.size() + b) = grad.col(b * unit_disp_ids.size() + a);
        }
    }
}

void State::homogenize_weighted_linear_elasticity_grad(Eigen::MatrixXd &C_H, Eigen::MatrixXd &grad)
{
    if (!mesh)
    {
        logger().error("Load the mesh first!");
        return;
    }
    if (n_bases <= 0)
    {
        logger().error("Build the bases first!");
        return;
    }
    if (formulation() != "LinearElasticity")
    {
        logger().error("Wrong formulation!");
        return;
    }
    if (!args["space"]["advanced"]["periodic_basis"])
    {
        logger().error("This homogenization only supports periodic basis!");
        return;
    }

    assemble_stiffness_mat();
    
    const int dim = mesh->dimension();
    const auto &gbases = geom_bases();
    
    std::vector<std::pair<int, int>> unit_disp_ids;
    if (dim == 2)
        unit_disp_ids = {{0, 0}, {1, 1}, {0, 1}};
    else
        unit_disp_ids = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

    Eigen::MatrixXd rhs; //, unit_disp;
    // unit_disp.setZero(stiffness.rows(), unit_disp_ids.size());
    rhs.setZero(stiffness.rows(), unit_disp_ids.size());

    const LameParameters &params = assembler.lame_params();
    Eigen::MatrixXd density_mat = params.density_mat_;
    
    std::vector<Eigen::MatrixXd> unit_strains(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));
    for (int id = 0; id < unit_disp_ids.size(); id++)
    {
        const auto &pair = unit_disp_ids[id];
        auto &unit_strain = unit_strains[id];

        Eigen::MatrixXd grad_unit(dim, dim);
        grad_unit.setZero();
        grad_unit(pair.first, pair.second) = 1;
        unit_strain = (grad_unit + grad_unit.transpose()) / 2;
    }

    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;

        for (int q = 0; q < quadrature.weights.size(); q++)
        {
            double lambda, mu;
            params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu, true);

            for (const auto &v : vals.basis_values)
            {
                for (int d = 0; d < dim; d++)
                {
                    Eigen::MatrixXd basis_strain, grad_basis;
                    basis_strain.setZero(dim, dim);
                    grad_basis.setZero(dim, dim);
                    grad_basis.row(d) = v.grad_t_m.row(q);
                    basis_strain = (grad_basis + grad_basis.transpose()) / 2; 

                    for (int k = 0; k < unit_disp_ids.size(); k++)
                    {
                        const auto &unit_strain = unit_strains[k];

                        const double value = quadrature.weights(q) * vals.det(q) * (2 * mu * (unit_strain.array() * basis_strain.array()).sum() + lambda * unit_strain.trace() * basis_strain.trace());

                        for (auto g : v.global)
                            rhs(g.index * dim + d, k) += value * g.val;
                    }
                }
            }
        }
    }

    // solve elastic problem
    StiffnessMatrix A = stiffness;
    const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
    int precond_num = problem_dim * n_bases;

    if (boundary_nodes.size() > 0)
    {
        logger().error("Homogenization with Dirichlet BC not implemented!");
        return;
    }

    apply_lagrange_multipliers(A);
    rhs.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), rhs.cols()));

    Eigen::MatrixXd w;
    w.setZero(rhs.rows(), rhs.cols());
    auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);
    auto A_tmp = A;
    {
        prefactorize(*solver, A_tmp, boundary_nodes, precond_num, args["output"]["data"]["stiffness_mat"]);
    }
    for (int k = 0; k < rhs.cols(); k++)
    {
        // auto A_tmp = A;
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        dirichlet_solve_prefactorized(*solver, A, b, boundary_nodes, x);
        // polysolve::dirichlet_solve(*solver, A_tmp, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
        // solver->getInfo(solver_info);
        w.col(k) = x;
    }

    auto error = (A_tmp * w - rhs).norm();
    if (std::isnan(error) || error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);
    
    w.conservativeResize(n_bases * dim, w.cols());

    // compute homogenized stiffness
    C_H.setZero(unit_disp_ids.size(), unit_disp_ids.size());
    
    RowVectorNd min, max;
    mesh->bounding_box(min, max);
    double volume = 1;
    for (int d = 0; d < min.size(); d++)
        volume *= (max(d) - min(d));
    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;

        for (int q = 0; q < quadrature.weights.size(); q++)
        {
            double lambda, mu;
            params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu, true);

            std::vector<Eigen::MatrixXd> react_strains(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));

            for (int id = 0; id < unit_disp_ids.size(); id++)
            {
                Eigen::MatrixXd grad_react(dim, dim);
                grad_react.setZero();
                for (const auto &v : vals.basis_values)
                    for (int d = 0; d < dim; d++)
                    {
                        double coeff = 0;
                        for (const auto &g : v.global)
                            coeff += w(g.index * dim + d, id) * g.val;
                        grad_react.row(d) += v.grad_t_m.row(q) * coeff;
                    }
                
                react_strains[id] = (grad_react + grad_react.transpose()) / 2;
            }

            for (int row_id = 0; row_id < unit_disp_ids.size(); row_id++)
            {
                Eigen::MatrixXd strain_diff_ij = unit_strains[row_id] - react_strains[row_id];

                for (int col_id = 0; col_id < unit_disp_ids.size(); col_id++)
                {
                    Eigen::MatrixXd strain_diff_kl = unit_strains[col_id] - react_strains[col_id];

                    const double value = 2 * mu * (strain_diff_ij.array() * strain_diff_kl.array()).sum() + lambda * strain_diff_ij.trace() * strain_diff_kl.trace();
                    C_H(row_id, col_id) += vals.quadrature.weights(q) * vals.det(q) * value;
                }
            }
        }
    }

    C_H /= volume;

    Eigen::MatrixXd adjoint;
    solve_adjoint_homogenize_linear_elasticity(w, adjoint);

    grad.setZero(bases.size(), unit_disp_ids.size() * unit_disp_ids.size());
    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;
        
        for (int q = 0; q < quadrature.weights.size(); q++)
        {
            double lambda, mu;
            params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu, false);
            
            auto weak_form = [lambda, mu](const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
            {
                return 2 * mu * (A.array() * B.array()).sum() + lambda * A.trace() * B.trace();
            };

            for (int a = 0; a < unit_disp_ids.size(); a++)
            {
                Eigen::MatrixXd grad_sol_a(dim, dim), grad_adjoint_a(dim, dim);
                grad_sol_a.setZero();
                grad_adjoint_a.setZero();
                for (const auto &v : vals.basis_values)
                    for (int d = 0; d < dim; d++)
                    {
                        double coeff = 0;
                        for (const auto &g : v.global)
                            coeff += w(g.index * dim + d, a) * g.val;

                        grad_sol_a.row(d) += v.grad_t_m.row(q) * coeff;

                        coeff = 0;
                        for (const auto &g : v.global)
                            coeff += adjoint(g.index * dim + d, a) * g.val;
                        grad_adjoint_a.row(d) += v.grad_t_m.row(q) * coeff;
                    }
                
                Eigen::MatrixXd react_strain_a, adjoint_strain_a;
                react_strain_a = (grad_sol_a + grad_sol_a.transpose()) / 2;
                auto diff_strain_a = react_strain_a - unit_strains[a];

                adjoint_strain_a = (grad_adjoint_a + grad_adjoint_a.transpose()) / 2;

                const double value1 = quadrature.weights(q) * vals.det(q) * weak_form(adjoint_strain_a, diff_strain_a);

                for (int b = 0; b <= a; b++)
                {
                    Eigen::MatrixXd grad_sol_b(dim, dim), grad_adjoint_b(dim, dim);
                    grad_sol_b.setZero();
                    grad_adjoint_b.setZero();
                    for (const auto &v : vals.basis_values)
                        for (int d = 0; d < dim; d++)
                        {
                            double coeff = 0;
                            for (const auto &g : v.global)
                                coeff += w(g.index * dim + d, b) * g.val;

                            grad_sol_b.row(d) += v.grad_t_m.row(q) * coeff;

                            coeff = 0;
                            for (const auto &g : v.global)
                                coeff += adjoint(g.index * dim + d, b) * g.val;
                            grad_adjoint_b.row(d) += v.grad_t_m.row(q) * coeff;
                        }
                    
                    Eigen::MatrixXd react_strain_b, adjoint_strain_b;
                    react_strain_b = (grad_sol_b + grad_sol_b.transpose()) / 2;
                    auto diff_strain_b = react_strain_b - unit_strains[b];

                    adjoint_strain_b = (grad_adjoint_b + grad_adjoint_b.transpose()) / 2;

                    const double value2 = quadrature.weights(q) * vals.det(q) * weak_form(adjoint_strain_b, diff_strain_b);
                    const double value3 = quadrature.weights(q) * vals.det(q)  * weak_form(diff_strain_b, diff_strain_a);

                    grad(e, a * unit_disp_ids.size() + b) += (value1 + value2 + value3) * pow(params.density(e), params.density_power_ - 1) * params.density_power_;
                    if (a != b)
                        grad(e, b * unit_disp_ids.size() + a) = grad(e, a * unit_disp_ids.size() + b);
                }
            }
        }
    }
    grad /= volume;
}

void State::homogenize_weighted_stokes(Eigen::MatrixXd &K_H)
{
    if (!mesh)
    {
        logger().error("Load the mesh first!");
        return;
    }
    if (n_bases <= 0)
    {
        logger().error("Build the bases first!");
        return;
    }
    if (formulation() != "Stokes")
    {
        logger().error("Wrong formulation!");
        return;
    }
    
    const int dim = mesh->dimension();
    const auto &gbases = geom_bases();

    // assemble stiffness
    {
        Density solid_density;
        Eigen::MatrixXd solid_density_mat = assembler.lame_params().density_mat_.array() - min_solid_density;
        assert(solid_density_mat.minCoeff() >= 0);
        solid_density.init_multimaterial(solid_density_mat);

        Density fluid_density;
        Eigen::MatrixXd fluid_density_mat = 1 - solid_density_mat.array();
        assert(fluid_density_mat.minCoeff() >= 0);
        fluid_density.init_multimaterial(fluid_density_mat);

        StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
        assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, fluid_density, bases, geom_bases(), ass_vals_cache, velocity_stiffness);
        assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, geom_bases(), pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
        assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, geom_bases(), pressure_ass_vals_cache, pressure_stiffness);

        assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, solid_density, bases, geom_bases(), ass_vals_cache, mass);

        AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, dim, use_avg_pressure ? assembler.is_fluid(formulation()) : false, velocity_stiffness + mass / args["materials"]["solid_permeability"].get<double>(), mixed_stiffness, pressure_stiffness, stiffness);
    }
    
    Eigen::MatrixXd rhs(stiffness.rows(), dim);
    rhs.setZero();
    
    // assemble unit test force rhs
    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

        const Quadrature &quadrature = vals.quadrature;

        const int n_loc_bases_ = int(vals.basis_values.size());
        for (int i = 0; i < n_loc_bases_; ++i)
        {
            const AssemblyValues &v = vals.basis_values[i];
            double rhs_value = (vals.det.array() * quadrature.weights.array() * v.val.array()).sum();

            for (int d = 0; d < dim; ++d)
                for (int ii = 0; ii < v.global.size(); ++ii)
                    rhs(v.global[ii].index * dim + d, d) += rhs_value * v.global[ii].val;
        }
    }
    
    // solve fluid problem
    StiffnessMatrix A = stiffness;
    const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
    int precond_num = problem_dim * n_bases;

	auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);
    auto A_tmp = A;
    {
        prefactorize(*solver, A_tmp, boundary_nodes, precond_num, args["output"]["data"]["stiffness_mat"]);
    }

    Eigen::MatrixXd w;
    w.setZero(rhs.rows(), rhs.cols());

    // solve for w: \sum_k w_{ji,kk} - p_{i,j} = -delta_{ji}
    // StiffnessMatrix A_tmp;
    for (int k = 0; k < dim; k++)
    {
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        dirichlet_solve_prefactorized(*solver, A, b, boundary_nodes, x);
        // A_tmp = A;
        // polysolve::dirichlet_solve(*solver, A_tmp, b, std::vector<int>(), x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], false, use_avg_pressure);
        // solver->getInfo(solver_info);
        w.col(k) = x;
    }

    auto res = A_tmp * w - rhs;

    const auto error = res.norm();
    if (std::isnan(error) || error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    w.conservativeResize(n_bases * dim, w.cols());

    for (int id = 0; id < w.cols(); id++)
    {       
        sol = w.col(id);
        pressure.setZero(n_pressure_bases, 1);
        save_vtu("homo_" + std::to_string(id) + ".vtu", 1.);
    }

    // compute homogenized permeability
    K_H.setZero(dim, dim);

    RowVectorNd min, max;
    mesh->bounding_box(min, max);
    double volume = 1;
    for (int d = 0; d < min.size(); d++)
        volume *= (max(d) - min(d));

    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;

        std::vector<Eigen::MatrixXd> values(quadrature.weights.size(), Eigen::MatrixXd::Zero(dim, dim));
        const int n_loc_bases = vals.basis_values.size();
		for (int l = 0; l < n_loc_bases; ++l)
		{
			const auto &val = vals.basis_values[l];
            for (size_t ii = 0; ii < val.global.size(); ++ii)
                for (int j = 0; j < dim; j++)
                    for (int i = 0; i < dim; i++)
                        for (int q = 0; q < values.size(); q++)
                            values[q](j, i) += val.global[ii].val * w(val.global[ii].index * dim + j, i) * val.val(q);
        }

        for (int q = 0; q < values.size(); q++)
            for (int j = 0; j < dim; j++)
                for (int i = 0; i < dim; i++)
                    K_H(i, j) += values[q](i, j) * vals.det(q) * quadrature.weights(q);
    }

    // auto velocity_block = stiffness.topLeftCorner(n_bases * dim, n_bases * dim);
    // for (int i = 0; i < dim; i++)
    // {
    //     Eigen::VectorXd tmp = velocity_block * w.block(0, i, n_bases * dim, 1);
    //     for (int j = 0; j < dim; j++)
    //         K_H(i, j) = (tmp.array() * w.block(0, j, n_bases * dim, 1).array()).sum();
    // }

    K_H /= volume;
}

void State::homogenize_weighted_stokes_grad(Eigen::MatrixXd &K_H, Eigen::MatrixXd &grad)
{
    if (!mesh)
    {
        logger().error("Load the mesh first!");
        return;
    }
    if (n_bases <= 0)
    {
        logger().error("Build the bases first!");
        return;
    }
    if (formulation() != "Stokes")
    {
        logger().error("Wrong formulation!");
        return;
    }
    
    const int dim = mesh->dimension();
    const auto &gbases = geom_bases();

    // assemble stiffness
    {
        Density solid_density;
        Eigen::MatrixXd solid_density_mat = assembler.lame_params().density_mat_.array() - min_solid_density;
        assert(solid_density_mat.minCoeff() >= 0);
        solid_density.init_multimaterial(solid_density_mat);

        Density fluid_density;
        Eigen::MatrixXd fluid_density_mat = 1 - solid_density_mat.array();
        assert(fluid_density_mat.minCoeff() >= 0);
        fluid_density.init_multimaterial(fluid_density_mat);

        StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
        assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, fluid_density, bases, geom_bases(), ass_vals_cache, velocity_stiffness);
        assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, geom_bases(), pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
        assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, geom_bases(), pressure_ass_vals_cache, pressure_stiffness);

        assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, solid_density, bases, geom_bases(), ass_vals_cache, mass);

        AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, dim, use_avg_pressure ? assembler.is_fluid(formulation()) : false, velocity_stiffness + mass / args["materials"]["solid_permeability"].get<double>(), mixed_stiffness, pressure_stiffness, stiffness);
    }
    
    Eigen::MatrixXd rhs;
    rhs.setZero(stiffness.rows(), dim);
    
    // assemble unit test force rhs
    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

        const Quadrature &quadrature = vals.quadrature;

        const int n_loc_bases_ = int(vals.basis_values.size());
        for (int i = 0; i < n_loc_bases_; ++i)
        {
            const AssemblyValues &v = vals.basis_values[i];
            double rhs_value = (vals.det.array() * quadrature.weights.array() * v.val.array()).sum();

            for (int d = 0; d < dim; ++d)
                for (int ii = 0; ii < v.global.size(); ++ii)
                    rhs(v.global[ii].index * dim + d, d) += rhs_value * v.global[ii].val;
        }
    }
    
    // solve fluid problem
    const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
    int precond_num = problem_dim * n_bases;

	auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);
    auto A = stiffness;
    {
        prefactorize(*solver, A, std::vector<int>(), precond_num, args["output"]["data"]["stiffness_mat"]);
    }

    Eigen::MatrixXd w;
    w.setZero(rhs.rows(), rhs.cols());

    // solve for w: \sum_k w_{ji,kk} - p_{i,j} = -delta_{ji}
    // StiffnessMatrix A_tmp;
    for (int k = 0; k < dim; k++)
    {
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        dirichlet_solve_prefactorized(*solver, stiffness, b, std::vector<int>(), x);
        // A_tmp = A;
        // polysolve::dirichlet_solve(*solver, A_tmp, b, std::vector<int>(), x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], false, use_avg_pressure);
        // solver->getInfo(solver_info);
        w.col(k) = x;
    }

    auto res = A * w - rhs;

    const auto error = res.norm();
    if (std::isnan(error) || error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    for (int id = 0; id < w.cols(); id++)
    {       
        sol = w.block(0, id, n_bases * dim, 1);
        pressure = w.block(n_bases * dim, id, n_pressure_bases, 1);
        save_vtu("homo_" + std::to_string(id) + ".vtu", 1.);
    }

    w.conservativeResize(n_bases * dim, w.cols());

    // compute homogenized permeability
    K_H.setZero(dim, dim);

    RowVectorNd min, max;
    mesh->bounding_box(min, max);
    double volume = 1;
    for (int d = 0; d < min.size(); d++)
        volume *= (max(d) - min(d));

    // auto velocity_block = stiffness.topLeftCorner(n_bases * dim, n_bases * dim);
    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;

        std::vector<Eigen::MatrixXd> values(quadrature.weights.size(), Eigen::MatrixXd::Zero(dim, dim));
        const int n_loc_bases = vals.basis_values.size();
		for (int l = 0; l < n_loc_bases; ++l)
		{
			const auto &val = vals.basis_values[l];
            for (size_t ii = 0; ii < val.global.size(); ++ii)
                for (int j = 0; j < dim; j++)
                    for (int i = 0; i < dim; i++)
                        for (int q = 0; q < values.size(); q++)
                            values[q](j, i) += val.global[ii].val * w(val.global[ii].index * dim + j, i) * val.val(q);
        }

        for (int q = 0; q < values.size(); q++)
            for (int j = 0; j < dim; j++)
                for (int i = 0; i < dim; i++)
                    K_H(i, j) += values[q](i, j) * vals.det(q) * quadrature.weights(q);
    }

    K_H /= volume;

    // J = avg(K_ii), assemble -grad_w J
    rhs.setZero(stiffness.rows(), dim);
    // for (int d = 0; d < dim; d++)
    // {
    //     rhs.block(0, d, w.rows(), 1) = -2 * velocity_block * w.col(d);
    // }
    {
        for (int e = 0; e < bases.size(); e++)
        {
            ElementAssemblyValues vals;
            // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
            ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

            const Quadrature &quadrature = vals.quadrature;

            for (const auto &v : vals.basis_values)
            {
                for (const auto &g : v.global)
                {
                    for (int d1 = 0; d1 < dim; d1++)
                        rhs(g.index * dim + d1, d1) -= g.val * (quadrature.weights.array() * vals.det.array() * v.val.array()).sum();
                }
            }
        }
    }

    // adjoint solve
    Eigen::MatrixXd adjoint;
    adjoint.setZero(rhs.rows(), dim);

    for (int k = 0; k < dim; k++)
    {
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = adjoint.col(k);
        dirichlet_solve_prefactorized(*solver, stiffness, b, std::vector<int>(), x);
        // A_tmp = velocity_block;
        // polysolve::dirichlet_solve(*adjoint_solver, A_tmp, b, std::vector<int>(), x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], false, use_avg_pressure);
        // solver->getInfo(solver_info);
        adjoint.col(k) = x;
    }

    auto adjoint_res = A * adjoint - rhs;

    const auto adjoint_error = adjoint_res.norm();
    if (std::isnan(error) || error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    adjoint.conservativeResize(n_bases * dim, dim);
    
    // compute grad using adjoint
    grad.setZero(bases.size(), dim * dim);
    for (int k1 = 0; k1 < dim; k1++)
    for (int k2 = 0; k2 < dim; k2++)
    {
        for (int e = 0; e < bases.size(); e++)
        {
            ElementAssemblyValues vals;
            // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
            ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

            const Quadrature &quadrature = vals.quadrature;

            Eigen::MatrixXd grad_adjoint, grad_sol;
            Eigen::MatrixXd val_adjoint, val_sol;

            for (int q = 0; q < quadrature.weights.size(); q++)
            {
                grad_adjoint.setZero(dim, dim);
                grad_sol.setZero(dim, dim);
                val_adjoint.setZero(dim, 1);
                val_sol.setZero(dim, 1);

                for (const auto &v : vals.basis_values)
                    for (int ii = 0; ii < v.global.size(); ii++)
                        for (int d = 0; d < dim; d++)
                        {
                            grad_sol.row(d) += v.grad_t_m.row(q) * v.global[ii].val * w(v.global[ii].index * dim + d, k1);
                            grad_adjoint.row(d) += v.grad_t_m.row(q) * v.global[ii].val * adjoint(v.global[ii].index * dim + d, k2);

                            val_sol(d) += v.val(q) * v.global[ii].val * w(v.global[ii].index * dim + d, k1);
                            val_adjoint(d) += v.val(q) * v.global[ii].val * adjoint(v.global[ii].index * dim + d, k2);
                        }
                
                const double value1 = (grad_sol.array() * grad_adjoint.array()).sum() - (val_sol.array() * val_adjoint.array()).sum() / args["materials"]["solid_permeability"].get<double>();
                const double value2 = 0; // (grad_sol.array() * grad_sol.array()).sum() - (val_sol.array() * val_sol.array()).sum() / args["materials"]["solid_permeability"].get<double>();
                grad(e, k1 * dim + k2) += (value1 + value2) * quadrature.weights(q) * vals.det(q);
            }
        }
    }

    grad /= -volume;
}

}