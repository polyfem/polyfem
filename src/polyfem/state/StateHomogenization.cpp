#include <polyfem/State.hpp>

#include <polyfem/utils/StringUtils.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

namespace polyfem {

using namespace assembler;
using namespace mesh;
using namespace solver;
using namespace utils;
using namespace quadrature;

void State::homogenize_linear_elasticity(Eigen::MatrixXd &C_H)
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
    if (stiffness.rows() == 0)
    {
        assemble_stiffness_mat();
    }
    
    const int dim = mesh->dimension();
    const auto &gbases = iso_parametric() ? bases : geom_bases;
    
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

    // fix translations
    int n_lagrange_multiplier = 0;
    {
        logger().info("Pure periodic boundary condition, use Lagrange multiplier to find unique solution...");
        
        n_lagrange_multiplier = remove_pure_periodic_singularity(A);
        rhs.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), rhs.cols()));
    }

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

    const auto error = (A_tmp * w - rhs).norm() / rhs.norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    // auto diff = unit_disp - w;
    // for (int id = 0; id < unit_disp_ids.size(); id++)
    // {       
    //     sol = diff.col(id);
    //     save_vtu("homo_" + std::to_string(unit_disp_ids[id].first) + std::to_string(unit_disp_ids[id].second) + ".vtu", 1.);
    // }

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
                    C_H(row_id, col_id) += quadrature.weights(q) * vals.det(q) * value;
                }
            }
        }
    }

    C_H /= volume;
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

    assemble_stiffness_mat();
    
    const int dim = mesh->dimension();
    const auto &gbases = iso_parametric() ? bases : geom_bases;
    
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

                        const double value = quadrature.weights(q) * vals.det(q) * density(quadrature.points.row(q), vals.val.row(q), vals.element_id) * (2 * mu * (unit_strain.array() * basis_strain.array()).sum() + lambda * unit_strain.trace() * basis_strain.trace());

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

    // fix translations
    int n_lagrange_multiplier = 0;
    {
        logger().info("Pure periodic boundary condition, use Lagrange multiplier to find unique solution...");
        
        n_lagrange_multiplier = remove_pure_periodic_singularity(A);
        rhs.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), rhs.cols()));
    }

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

    const auto error = (A_tmp * w - rhs).norm() / rhs.norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);
    
    w.conservativeResize(n_bases * dim, w.cols());

    // auto diff = unit_disp - w;
    // for (int id = 0; id < unit_disp_ids.size(); id++)
    // {       
    //     sol = diff.col(id);
    //     save_vtu("homo_" + std::to_string(unit_disp_ids[id].first) + std::to_string(unit_disp_ids[id].second) + ".vtu", 1.);
    // }

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
                    C_H(row_id, col_id) += vals.quadrature.weights(q) * vals.det(q) * density(quadrature.points.row(q), vals.val.row(q), vals.element_id) * value;
                }
            }
        }
    }

    C_H /= volume;
}

void State::homogenize_weighted_linear_elasticity_grad(Eigen::MatrixXd &C_H, Eigen::VectorXd &grad)
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

    assemble_stiffness_mat();
    
    const int dim = mesh->dimension();
    const auto &gbases = iso_parametric() ? bases : geom_bases;
    
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

    // fix translations
    int n_lagrange_multiplier = 0;
    {
        logger().info("Pure periodic boundary condition, use Lagrange multiplier to find unique solution...");
        
        n_lagrange_multiplier = remove_pure_periodic_singularity(A);
        rhs.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), rhs.cols()));
    }

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

    auto error = (A_tmp * w - rhs).norm() / rhs.norm();
    if (error > 1e-4)
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

    // assemble -dJdu
    rhs.setZero(stiffness.rows(), unit_disp_ids.size());
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
                            rhs(g.index * dim + d, id) -= value * g.val;
                    }
                }
            }
        }
    }
    rhs *= 2;

    rhs.conservativeResizeLike(Eigen::MatrixXd::Zero(A.rows(), rhs.cols()));

    Eigen::MatrixXd adjoint;
    adjoint.setZero(rhs.rows(), rhs.cols());

    for (int k = 0; k < rhs.cols(); k++)
    {
        // auto A_tmp = A;
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = adjoint.col(k);
        dirichlet_solve_prefactorized(*solver, A, b, boundary_nodes, x);
        adjoint.col(k) = x;
    }

    error = (A_tmp * adjoint - rhs).norm() / rhs.norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);
    
    adjoint.conservativeResize(n_bases * dim, adjoint.cols());

    grad.setZero(bases.size(), 1);
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
            
            for (int id = 0; id < unit_disp_ids.size(); id++)
            {
                Eigen::MatrixXd grad_sol(dim, dim), grad_adjoint(dim, dim);
                grad_sol.setZero();
                grad_adjoint.setZero();
                for (const auto &v : vals.basis_values)
                    for (int d = 0; d < dim; d++)
                    {
                        double coeff = 0;
                        for (const auto &g : v.global)
                            coeff += w(g.index * dim + d, id) * g.val;

                        grad_sol.row(d) += v.grad_t_m.row(q) * coeff;

                        coeff = 0;
                        for (const auto &g : v.global)
                            coeff += adjoint(g.index * dim + d, id) * g.val;
                        grad_adjoint.row(d) += v.grad_t_m.row(q) * coeff;
                    }
                
                Eigen::MatrixXd react_strain, adjoint_strain;
                react_strain = (grad_sol + grad_sol.transpose()) / 2;
                auto diff_strain = react_strain - unit_strains[id];

                adjoint_strain = (grad_adjoint + grad_adjoint.transpose()) / 2;
                
                const double value1 = quadrature.weights(q) * vals.det(q) * (2 * mu * (diff_strain.array() * adjoint_strain.array()).sum() + lambda * diff_strain.trace() * adjoint_strain.trace());

                const double value2 = quadrature.weights(q) * vals.det(q) * (2 * mu * (diff_strain.array() * diff_strain.array()).sum() + lambda * diff_strain.trace() * diff_strain.trace());

                grad(e) += value1 + value2;
            }
        }
    }
    grad /= volume;
}

void State::homogenize_stokes(Eigen::MatrixXd &K_H)
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
    if (stiffness.rows() == 0)
    {
        assemble_stiffness_mat();
    }
    
    const int dim = mesh->dimension();
    const auto &gbases = iso_parametric() ? bases : geom_bases;
    
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

    if (boundary_nodes.size() == 0)
        logger().error("Homogenization of Stokes should have dirichlet boundary!");

	auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);
    auto A_tmp = A;
    {
        prefactorize(*solver, A_tmp, boundary_nodes, precond_num, args["output"]["data"]["stiffness_mat"]);
    }

    Eigen::MatrixXd w;
    w.setZero(rhs.rows(), rhs.cols());

    // dirichlet bc
    for (int b : boundary_nodes)
        rhs.row(b).setZero();

    // solve for w: \sum_k w_{ji,kk} - p_{i,j} = -delta_{ji}
    for (int k = 0; k < dim; k++)
    {
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        dirichlet_solve_prefactorized(*solver, A, b, boundary_nodes, x);
        // A_tmp = A;
        // polysolve::dirichlet_solve(*solver, A_tmp, b, boundary_nodes, x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], false, use_avg_pressure);
        // solver->getInfo(solver_info);
        w.col(k) = x;
    }

    auto res = A_tmp * w - rhs;

    const auto error = res.norm() / rhs.norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    w.conservativeResize(n_bases * dim, w.cols());

    for (int id = 0; id < w.cols(); id++)
    {       
        sol = w.block(0, id, n_bases * dim, 1);
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

    // for (int e = 0; e < bases.size(); e++)
    // {
    //     ElementAssemblyValues vals;
    //     // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
    //     ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

    //     const Quadrature &quadrature = vals.quadrature;

    //     std::vector<Eigen::MatrixXd> values(quadrature.weights.size(), Eigen::MatrixXd::Zero(dim, dim));
    //     const int n_loc_bases = vals.basis_values.size();
	// 	for (int l = 0; l < n_loc_bases; ++l)
	// 	{
	// 		const auto &val = vals.basis_values[l];
    //         for (size_t ii = 0; ii < val.global.size(); ++ii)
    //             for (int j = 0; j < dim; j++)
    //                 for (int i = 0; i < dim; i++)
    //                     for (int q = 0; q < values.size(); q++)
    //                         values[q](j, i) += val.global[ii].val * w(val.global[ii].index * dim + j, i) * val.val(q);
    //     }

    //     for (int q = 0; q < values.size(); q++)
    //         for (int j = 0; j < dim; j++)
    //             for (int i = 0; i < dim; i++)
    //                 K_H(i, j) += values[q](i, j) * vals.det(q) * quadrature.weights(q);
    // }

    auto velocity_block = stiffness.topLeftCorner(n_bases * dim, n_bases * dim);
    for (int i = 0; i < dim; i++)
    {
        Eigen::VectorXd tmp = velocity_block * w.block(0, i, n_bases * dim, 1);
        for (int j = 0; j < dim; j++)
            K_H(i, j) = (tmp.array() * w.block(0, j, n_bases * dim, 1).array()).sum();
    }

    K_H /= volume;
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
    if (!density.is_mat())
    {
        logger().error("Only per element density is supported in homogenization!");
        return;
    }
    
    const int dim = mesh->dimension();
    const auto &gbases = iso_parametric() ? bases : geom_bases;

    // assemble stiffness
    Density solid_density;
    {
        StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
        assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, density, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, velocity_stiffness);
        assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, iso_parametric() ? bases : geom_bases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
        assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, iso_parametric() ? bases : geom_bases, pressure_ass_vals_cache, pressure_stiffness);

        Eigen::MatrixXd solid_density_mat;
        density.get_multimaterial(solid_density_mat);
        solid_density_mat = 1 - solid_density_mat.array();

        assert(solid_density_mat.minCoeff() >= 0);
        solid_density.init_multimaterial(solid_density_mat);

        assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, solid_density, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, mass);

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

    const auto error = res.norm() / rhs.norm();
    if (error > 1e-4)
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

    // for (int e = 0; e < bases.size(); e++)
    // {
    //     ElementAssemblyValues vals;
    //     // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
    //     ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

    //     const Quadrature &quadrature = vals.quadrature;

    //     std::vector<Eigen::MatrixXd> values(quadrature.weights.size(), Eigen::MatrixXd::Zero(dim, dim));
    //     const int n_loc_bases = vals.basis_values.size();
	// 	for (int l = 0; l < n_loc_bases; ++l)
	// 	{
	// 		const auto &val = vals.basis_values[l];
    //         for (size_t ii = 0; ii < val.global.size(); ++ii)
    //             for (int j = 0; j < dim; j++)
    //                 for (int i = 0; i < dim; i++)
    //                     for (int q = 0; q < values.size(); q++)
    //                         values[q](j, i) += val.global[ii].val * w(val.global[ii].index * dim + j, i) * val.val(q);
    //     }

    //     for (int q = 0; q < values.size(); q++)
    //         for (int j = 0; j < dim; j++)
    //             for (int i = 0; i < dim; i++)
    //                 K_H(i, j) += values[q](i, j) * vals.det(q) * quadrature.weights(q);
    // }

    auto velocity_block = stiffness.topLeftCorner(n_bases * dim, n_bases * dim);
    for (int i = 0; i < dim; i++)
    {
        Eigen::VectorXd tmp = velocity_block * w.block(0, i, n_bases * dim, 1);
        for (int j = 0; j < dim; j++)
            K_H(i, j) = (tmp.array() * w.block(0, j, n_bases * dim, 1).array()).sum();
    }

    K_H /= volume;
}

void State::homogenize_weighted_stokes_grad(Eigen::MatrixXd &K_H, Eigen::VectorXd &grad)
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
    if (!density.is_mat())
    {
        logger().error("Only per element density is supported in homogenization!");
        return;
    }
    
    const int dim = mesh->dimension();
    const auto &gbases = iso_parametric() ? bases : geom_bases;

    // assemble stiffness
    Density solid_density;
    {
        StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
        assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, density, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, velocity_stiffness);
        assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, iso_parametric() ? bases : geom_bases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
        assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, iso_parametric() ? bases : geom_bases, pressure_ass_vals_cache, pressure_stiffness);

        Eigen::MatrixXd solid_density_mat;
        density.get_multimaterial(solid_density_mat);
        solid_density_mat = 1 - solid_density_mat.array();

        assert(solid_density_mat.minCoeff() >= 0);
        solid_density.init_multimaterial(solid_density_mat);

        assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, solid_density, bases, iso_parametric() ? bases : geom_bases, ass_vals_cache, mass);

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

    const auto error = res.norm() / rhs.norm();
    if (error > 1e-4)
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

    auto velocity_block = stiffness.topLeftCorner(n_bases * dim, n_bases * dim);
    for (int i = 0; i < dim; i++)
    {
        Eigen::VectorXd tmp = velocity_block * w.col(i);
        for (int j = 0; j < dim; j++)
            K_H(i, j) = (tmp.array() * w.col(j).array()).sum();
    }

    K_H /= volume;

    // J = avg(K_ii), assemble -grad_w J
    rhs.setZero(stiffness.rows(), dim);
    for (int d = 0; d < dim; d++)
    {
        rhs.block(0, d, w.rows(), 1) = -2 * velocity_block * w.col(d);
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

    const auto adjoint_error = res.norm() / rhs.norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    adjoint.conservativeResize(n_bases * dim, dim);
    
    // compute grad using adjoint
    grad.setZero(bases.size(), 1);
    for (int k = 0; k < dim; k++)
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
                            grad_sol.row(d) += v.grad_t_m.row(q) * v.global[ii].val * w(v.global[ii].index * dim + d, k);
                            grad_adjoint.row(d) += v.grad_t_m.row(q) * v.global[ii].val * adjoint(v.global[ii].index * dim + d, k);

                            val_sol(d) += v.val(q) * v.global[ii].val * w(v.global[ii].index * dim + d, k);
                            val_adjoint(d) += v.val(q) * v.global[ii].val * adjoint(v.global[ii].index * dim + d, k);
                        }
                
                const double value1 = (grad_sol.array() * grad_adjoint.array()).sum() - (val_sol.array() * val_adjoint.array()).sum() / args["materials"]["solid_permeability"].get<double>();
                const double value2 = (grad_sol.array() * grad_sol.array()).sum() - (val_sol.array() * val_sol.array()).sum() / args["materials"]["solid_permeability"].get<double>();
                grad(e) += (value1 + value2) * quadrature.weights(q) * vals.det(q);
            }
        }
    }

    grad /= volume * dim;
}

}