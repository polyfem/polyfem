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

void State::homogenize_linear_elasticity()
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

    const LameParameters &params = assembler.lame_params();

    // assemble unit test force rhs
    for (int e = 0; e < bases.size(); e++)
    {
        const auto &bs = bases[e];
        ElementAssemblyValues vals;
        vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

        const Quadrature &quadrature = vals.quadrature;

        const int n_loc_bases_ = int(vals.basis_values.size());
        for (int i = 0; i < n_loc_bases_; ++i)
        {
            const AssemblyValues &v = vals.basis_values[i];
            // double rhs_value = (vals.det.array() * quadrature.weights.array() * v.val.array()).sum();

            int idx = 0;
            for (int k = 0; k < dim; k++)
            {
                for (int l = 0; l < dim; l++)
                {
                    if (k < l)
                        continue;
                    
                    Eigen::MatrixXd unit_strain(dim, dim);
                    unit_strain.setZero();
                    unit_strain(k, l) = 1;
                    unit_strain = 0.5 * (unit_strain + unit_strain.transpose());

                    for (int d = 0; d < dim; d++)
                    {
                        double rhs_value = 0;
                        for (int q = 0; q < quadrature.weights.size(); q++)
                        {
                            Eigen::MatrixXd test_strain(dim, dim);
                            test_strain.setZero();
                            test_strain.row(d) = v.grad_t_m.row(q);
                            test_strain = 0.5 * (test_strain + test_strain.transpose());
                            
                            double lambda, mu;
                            params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);
                            
                            double value = 2 * mu * (test_strain.array() * unit_strain.array()).sum() + lambda * test_strain.trace() * unit_strain.trace();
                            rhs_value += value * quadrature.weights(q) * vals.det(q);
                        }

                        for (const auto &g : v.global)
                            rhs(g.index * dim + d, idx) += rhs_value * g.val;
                    }

                    idx++;
                }
            }
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
            for (int k = 0; k < rhs.rows(); k++)
                rhs_periodic(index_map(k), l) += rhs(k, l);

        std::swap(rhs, rhs_periodic);
    }

    // fix rigid transformation
    const int n_extra_constraints = remove_pure_neumann_singularity(A);
    rhs.conservativeResizeLike(Eigen::VectorXd::Zero(A.rows(), rhs.cols()));

	auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);

    Eigen::MatrixXd w(rhs.rows(), rhs.cols());

    for (int k = 0; k < rhs.cols(); k++)
    {
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        polysolve::dirichlet_solve(*solver, A, b, boundary_nodes_tmp, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
        solver->getInfo(solver_info);
        w.col(k) = x;
    }

    const auto error = (A * w - rhs).norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);
    
    // periodic boundary handling
    {
        Eigen::MatrixXd tmp;
        tmp.setZero(n_bases * dim, w.cols());
        for (int i = 0; i < n_bases * dim; i++)
            tmp(i) = w(periodic_reduce_map(i));

        std::swap(tmp, w);
    }

    // compute homogenized stiffness
    Eigen::MatrixXd C_H(dim * dim, dim * dim);
    C_H.setZero();
    double volume = 0;
    for (int e = 0; e < bases.size(); e++)
    {
        ElementAssemblyValues vals;
        vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

        const Quadrature &quadrature = vals.quadrature;

        volume += (quadrature.weights.array() * vals.det.array()).sum();

        for (int q = 0; q < quadrature.weights.size(); q++)
        {
            double lambda, mu;
            params.lambda_mu(quadrature.points.row(q), vals.val.row(q), e, lambda, mu);

            for (int i = 0, row_id = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                {
                    if (i < j)
                        continue;

                    Eigen::MatrixXd unit_strain_ij(dim, dim);
                    unit_strain_ij.setZero();
                    unit_strain_ij(i, j) = 1;
                    unit_strain_ij = (unit_strain_ij + unit_strain_ij.transpose()) / 2;

                    Eigen::MatrixXd react_strain_ij(dim, dim);
                    react_strain_ij.setZero();
                    for (int b = 0; b < vals.basis_values.size(); b++)
                        for (int ii = 0; ii < vals.basis_values[b].global.size(); ii++)
                            for (int d = 0; d < dim; d++)
                                react_strain_ij.row(d) += vals.basis_values[b].grad_t_m.row(q) * w(vals.basis_values[b].global[ii].index * dim + d, row_id) * vals.basis_values[b].global[ii].val;
                    react_strain_ij = (react_strain_ij + react_strain_ij.transpose()) / 2;

                    Eigen::MatrixXd strain_ij = unit_strain_ij - react_strain_ij;

                    for (int k = 0, col_id = 0; k < dim; k++)
                        for (int l = 0; l < dim; l++)
                        {
                            if (k < l)
                                continue;

                            Eigen::MatrixXd unit_strain_kl(dim, dim);
                            unit_strain_kl.setZero();
                            unit_strain_kl(k, l) = 1;
                            unit_strain_kl = (unit_strain_kl + unit_strain_kl.transpose()) / 2;

                            Eigen::MatrixXd react_strain_kl(dim, dim);
                            react_strain_kl.setZero();
                            for (int b = 0; b < vals.basis_values.size(); b++)
                                for (int ii = 0; ii < vals.basis_values[b].global.size(); ii++)
                                    for (int d = 0; d < dim; d++)
                                        react_strain_kl.row(d) += vals.basis_values[b].grad_t_m.row(q) * w(vals.basis_values[b].global[ii].index * dim + d, row_id) * vals.basis_values[b].global[ii].val;
                            react_strain_kl = (react_strain_kl + react_strain_kl.transpose()) / 2;

                            Eigen::MatrixXd strain_kl = unit_strain_kl - react_strain_kl;

                            const double value = 2 * mu * (strain_ij.array() * strain_kl.array()).sum() + lambda * strain_ij.trace() * strain_kl.trace();
                            C_H(i * dim + j, k * dim + l) += vals.quadrature.weights(q) * vals.det(q) * value;

                            col_id++;
                        }
                    row_id++;
                }
        }
    }

    C_H /= volume;
}

void State::homogenize_stokes()
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
        const auto &bs = bases[e];
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
            for (int k = 0; k < rhs.rows(); k++)
                rhs_periodic(index_map(k), l) += rhs(k, l);

        std::swap(rhs, rhs_periodic);
    }

    if (boundary_nodes_tmp.size() == 0)
        logger().error("Homogenization of Stokes should have dirichlet boundary!");

	auto solver = polysolve::LinearSolver::create(args["solver"]["linear"]["solver"], args["solver"]["linear"]["precond"]);
    solver->setParameters(args["solver"]["linear"]);

    Eigen::MatrixXd w(rhs.rows(), rhs.cols());

    // solve for w: \sum_k w_{ji,kk} - p_{i,j} = -delta_{ji}
    for (int k = 0; k < dim; k++)
    {
        Eigen::VectorXd b = rhs.col(k);
        Eigen::VectorXd x = w.col(k);
        polysolve::dirichlet_solve(*solver, A, b, boundary_nodes_tmp, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
        solver->getInfo(solver_info);
        w.col(k) = x;
    }

    const auto error = (A * w - rhs).norm();
    if (error > 1e-4)
        logger().error("Solver error: {}", error);
    else
        logger().debug("Solver error: {}", error);

    // periodic boundary handling
    {
        Eigen::MatrixXd tmp;
        tmp.setZero(n_bases * dim, w.cols());
        for (int i = 0; i < n_bases * dim; i++)
            tmp(i) = w(periodic_reduce_map(i));

        std::swap(tmp, w);
    }

    // compute homogenized permeability
    Eigen::MatrixXd K_H(dim, dim);
    K_H.setZero();
    double volume = 0;
    for (int e = 0; e < bases.size(); e++)
    {
        const auto &bs = bases[e];
        ElementAssemblyValues vals;
        vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

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

        volume += (vals.det.array() * quadrature.weights.array()).sum();
    }

    K_H /= volume;
}

}