#include <polyfem/State.hpp>

#include <polyfem/StringUtils.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

namespace polyfem {

void State::homogenize_linear_elasticity()
{
    // assemble unit test strain rhs

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
    for (int k = 0; k < dim; k++)
    {
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
                for (int d = 0; d < dim; ++d)
                {
                    double rhs_value = 0;
                    for (int q = 0; q < quadrature.weights.size(); q++)
                    {
                        rhs_value += (d == k ? 1.0 : 0.0) * vals.det(q) * quadrature.weights(q) * v.val(q);
                    }
                    for (int ii = 0; ii < v.global.size(); ++ii)
                    {
                        rhs(v.global[ii].index * dim + d) += rhs_value * v.global[ii].val;
                    }
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

    if (boundary_nodes_tmp.size() == 0)
        logger().error("Homogenization of Stokes should have dirichlet boundary!");

	auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);
	const json &params = solver_params();
    solver->setParameters(params);

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