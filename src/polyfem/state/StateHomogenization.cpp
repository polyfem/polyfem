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

    Eigen::MatrixXd rhs, unit_disp;
    unit_disp.setZero(stiffness.rows(), unit_disp_ids.size());
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

    for (int p = 0; p < nodes_position.rows(); p++)
    {
        for (int k = 0; k < unit_disp_ids.size(); k++)
        {
            unit_disp(p * dim + unit_disp_ids[k].first, k) = nodes_position(p, unit_disp_ids[k].second);
        }
    }

    rhs = stiffness * unit_disp;

    // solve elastic problem
    StiffnessMatrix A = stiffness;
    auto boundary_nodes_tmp = boundary_nodes;
    const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
    int precond_num = problem_dim * n_bases;

    // periodic boundary handling
    {
        const int independent_dof = periodic_reduce_map.maxCoeff() + 1;
        
        StiffnessMatrix A_periodic(independent_dof, independent_dof);
        precond_num = independent_dof;
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(A.nonZeros());
        for (int k = 0; k < A.outerSize(); k++)
        {
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
            {
                entries.emplace_back(periodic_reduce_map(it.row()), periodic_reduce_map(it.col()), it.value());
            }
        }
        A_periodic.setFromTriplets(entries.begin(),entries.end());

        std::swap(A_periodic, A);

        // rhs under periodic basis
        Eigen::MatrixXd rhs_periodic;
        rhs_periodic.setZero(independent_dof, rhs.cols());
        for (int l = 0; l < rhs.cols(); l++)
        {
            for (int k = 0; k < rhs.rows(); k++)
                rhs_periodic(periodic_reduce_map(k), l) += rhs(k, l);

			for (int k : boundary_nodes)
				rhs_periodic(periodic_reduce_map(k), l) = rhs(k, l);
        } 

        std::swap(rhs, rhs_periodic);
    }

    if (boundary_nodes.size() > 0)
    {
        logger().error("Homogenization with Dirichlet BC not implemented!");
        return;
    }

    // fix translations
    int n_lagrange_multiplier = 0;
    if (args["boundary_conditions"]["periodic_boundary"].get<bool>())
    {
        if (assembler.is_mixed(formulation()))
        {
            logger().error("Pure periodic without Dirichlet not supported for mixed formulation!");
            return;
        }
        else
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
        prefactorize(*solver, A_tmp, boundary_nodes_tmp, precond_num, args["output"]["data"]["stiffness_mat"]);
    }
    for (int k = 0; k < rhs.cols(); k++)
    {
        // auto A_tmp = A;
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        dirichlet_solve_prefactorized(*solver, A, b, boundary_nodes_tmp, x);
        // polysolve::dirichlet_solve(*solver, A_tmp, b, boundary_nodes_tmp, x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
        // solver->getInfo(solver_info);
        w.col(k) = x;
    }

    const auto error = (A_tmp * w - rhs).norm() / rhs.norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);
    
    // periodic boundary handling
    {
        Eigen::MatrixXd tmp;
        tmp.setZero(n_bases * dim, w.cols());
        for (int i = 0; i < n_bases * dim; i++)
            for (int k = 0; k < w.cols(); k++)
                tmp(i, k) = w(periodic_reduce_map(i), k);

        std::swap(tmp, w);
    }

    auto diff = unit_disp - w;
    for (int id = 0; id < unit_disp_ids.size(); id++)
    {       
        sol = diff.col(id);
        save_vtu("homo_" + std::to_string(unit_disp_ids[id].first) + std::to_string(unit_disp_ids[id].second) + ".vtu", 1.);
    }

    // compute homogenized stiffness
    C_H.setZero(unit_disp_ids.size(), unit_disp_ids.size());
    for (int i = 0; i < C_H.rows(); i++)
        for (int j = 0; j < C_H.cols(); j++)
            C_H(i, j) = diff.col(i).transpose() * stiffness * diff.col(j);
    
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

    //     for (int q = 0; q < quadrature.weights.size(); q++)
    //     {
    //         double lambda, mu;
    //         params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);

    //         std::vector<Eigen::MatrixXd> react_strains(unit_disp_ids.size(), Eigen::MatrixXd::Zero(dim, dim));

    //         for (int id = 0; id < unit_disp_ids.size(); id++)
    //         {
    //             Eigen::MatrixXd grad(dim, dim);
    //             grad.setZero();
    //             for (const auto &v : vals.basis_values)
    //                 for (int d = 0; d < dim; d++)
    //                 {
    //                     double coeff = 0;
    //                     for (const auto &g : v.global)
    //                         coeff += w(g.index * dim + d, id) * g.val;
    //                     grad.row(d) += v.grad_t_m.row(q) * coeff;
    //                 }
                
    //             react_strains[id] = (grad + grad.transpose()) / 2;
    //         }

    //         for (int row_id = 0; row_id < unit_disp_ids.size(); row_id++)
    //         {
    //             Eigen::MatrixXd strain_diff_ij = unit_strains[row_id] - react_strains[row_id];

    //             for (int col_id = 0; col_id < unit_disp_ids.size(); col_id++)
    //             {
    //                 Eigen::MatrixXd strain_diff_kl = unit_strains[col_id] - react_strains[col_id];

    //                 const double value = 2 * mu * (strain_diff_ij.array() * strain_diff_kl.array()).sum() + lambda * strain_diff_ij.trace() * strain_diff_kl.trace();
    //                 C_H(row_id, col_id) += vals.quadrature.weights(q) * vals.det(q) * value;
    //             }
    //         }
    //     }
    // }

    C_H /= volume;
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
    auto boundary_nodes_tmp = boundary_nodes;
    const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
    int precond_num = problem_dim * n_bases;

    // periodic boundary handling
    {
        // new index for boundary_nodes
        std::vector<int> boundary_nodes_periodic = boundary_nodes;
        {
            for (int i = 0; i < boundary_nodes_periodic.size(); i++)
            {
                boundary_nodes_periodic[i] = periodic_reduce_map(boundary_nodes_periodic[i]);
            }

            std::sort(boundary_nodes_periodic.begin(), boundary_nodes_periodic.end());
            auto it = std::unique(boundary_nodes_periodic.begin(), boundary_nodes_periodic.end());
            boundary_nodes_periodic.resize(std::distance(boundary_nodes_periodic.begin(), it));
        }

        std::swap(boundary_nodes_periodic, boundary_nodes_tmp);

        const int independent_dof = periodic_reduce_map.maxCoeff() + 1;
        
        auto index_map = [&](int id){
            if (id < periodic_reduce_map.size())
                return periodic_reduce_map(id);
            else
                return (int)(id + independent_dof - n_bases * problem_dim);
        };

        StiffnessMatrix A_periodic(index_map(A.rows()), index_map(A.cols()));
        precond_num = independent_dof;
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(A.nonZeros());
        for (int k = 0; k < A.outerSize(); k++)
        {
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
            {
                entries.emplace_back(index_map(it.row()), index_map(it.col()), it.value());
            }
        }
        A_periodic.setFromTriplets(entries.begin(),entries.end());

        std::swap(A_periodic, A);

        // rhs under periodic basis
        Eigen::MatrixXd rhs_periodic;
        rhs_periodic.setZero(index_map(rhs.rows()), rhs.cols());
        for (int l = 0; l < rhs.cols(); l++)
        {
            for (int k = 0; k < rhs.rows(); k++)
                rhs_periodic(index_map(k), l) += rhs(k, l);

			for (int k : boundary_nodes)
				rhs_periodic(index_map(k), l) = rhs(k, l);
        } 

        std::swap(rhs, rhs_periodic);
    }

    if (boundary_nodes_tmp.size() == 0)
        logger().error("Homogenization of Stokes should have dirichlet boundary!");

	auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);

    Eigen::MatrixXd w;
    w.setZero(rhs.rows(), rhs.cols());

    // dirichlet bc
    for (int b : boundary_nodes_tmp)
        rhs.row(b).setZero();

    // solve for w: \sum_k w_{ji,kk} - p_{i,j} = -delta_{ji}
    StiffnessMatrix A_tmp;
    for (int k = 0; k < dim; k++)
    {
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        A_tmp = A;
        polysolve::dirichlet_solve(*solver, A_tmp, b, boundary_nodes_tmp, x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], false, use_avg_pressure);
        solver->getInfo(solver_info);
        w.col(k) = x;
    }

    auto res = A_tmp * w - rhs;

    const auto error = res.norm() / rhs.norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    // periodic boundary handling
    {
        Eigen::MatrixXd tmp;
        tmp.setZero(n_bases * dim, w.cols());
        for (int i = 0; i < n_bases * dim; i++)
            for (int l = 0; l < w.cols(); l++)
                tmp(i, l) = w(periodic_reduce_map(i), l);

        std::swap(tmp, w);
    }

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

    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        // vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);
        ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);

        const Quadrature &quadrature = vals.quadrature;

        std::vector<Eigen::MatrixXd> values(quadrature.weights.size(), Eigen::MatrixXd::Zero(dim, dim));
        std::vector<Eigen::MatrixXd> grads(quadrature.weights.size(), Eigen::MatrixXd::Zero(dim*dim, dim));
        const int n_loc_bases = vals.basis_values.size();
		for (int l = 0; l < n_loc_bases; ++l)
		{
			const auto &val = vals.basis_values[l];
            for (size_t ii = 0; ii < val.global.size(); ++ii)
                for (int j = 0; j < dim; j++)
                    for (int i = 0; i < dim; i++)
                        for (int q = 0; q < values.size(); q++)
                        {
                            values[q](j, i) += val.global[ii].val * w(val.global[ii].index * dim + j, i) * val.val(q);
                            grads[q].row(j * dim + i) += val.global[ii].val * w(val.global[ii].index * dim + j, i) * val.grad_t_m.row(q);
                        }
        }

        for (int q = 0; q < values.size(); q++)
            for (int j = 0; j < dim; j++)
                for (int i = 0; i < dim; i++)
                    for (int k = 0; k < dim; k++)
                    {
                        K_H(i, j) += (grads[q].row(k * dim + i).array() * grads[q].row(k * dim + j).array()).sum() * vals.det(q) * quadrature.weights(q);
                    }
    }

    K_H /= volume;
}

}