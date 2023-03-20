#include "NLHomoProblem.hpp"
#include <polyfem/io/Evaluator.hpp>

namespace polyfem::solver
{
    NLHomoProblem::NLHomoProblem(const int full_size,
                const std::vector<int> &boundary_nodes,
                const std::vector<mesh::LocalBoundary> &local_boundary,
                const int n_boundary_samples,
                const assembler::RhsAssembler &rhs_assembler,
                const State &state,
                const double t, const std::vector<std::shared_ptr<Form>> &forms) : NLProblem(full_size, boundary_nodes, local_boundary, n_boundary_samples, rhs_assembler, state, t, forms)
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();

        Eigen::MatrixXd X = io::Evaluator::get_bases_position(state_.n_bases, state_.mesh_nodes);

        constraint_grad.setZero(dim * dim, full_size_);
        for (int i = 0; i < X.rows(); i++)
            for (int j = 0; j < dim; j++)
                for (int k = 0; k < dim; k++)
                    constraint_grad(j * dim + k, i * dim + j) = X(i, k);
    }

    void NLHomoProblem::set_only_symmetric()
    {
        only_symmetric = true;

        const int dim = state_.mesh->dimension();

        symmetric_to_full.setZero(dim * dim, (dim * (dim + 1)) / 2);
        full_to_symmetric.setZero(dim * dim, (dim * (dim + 1)) / 2);
        for (int i = 0, idx = 0; i < dim; i++)
        {
            for (int j = i; j < dim; j++)
            {
                full_to_symmetric(i * dim + j, idx) = 1;
                
                symmetric_to_full(j * dim + i, idx) = 1;
                symmetric_to_full(i * dim + j, idx) = 1;
                
                idx++;
            }
        }
    }

    void NLHomoProblem::set_fixed_entry(const std::vector<int> &fixed_entry)
    {
        fixed_entry_ = fixed_entry;
    }

    void NLHomoProblem::full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();

        Eigen::MatrixXd tmp = macro_full_to_reduced_grad(constraint_grad);
        Eigen::MatrixXd A12 = full * tmp.transpose();
        Eigen::MatrixXd A22 = tmp * A12;

        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(full.nonZeros() + A12.size() * 2 + A22.size());

        for (int k = 0; k < full.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(full, k); it; ++it)
                entries.emplace_back(it.row(), it.col(), it.value());

        for (int i = 0; i < A12.rows(); i++)
            for (int j = 0; j < A12.cols(); j++)
            {
                entries.emplace_back(i, full.cols() + j, A12(i, j));
                entries.emplace_back(full.rows() + j, i, A12(i, j));
            }
        
        for (int i = 0; i < A22.rows(); i++)
            for (int j = 0; j < A22.cols(); j++)
                entries.emplace_back(i + full.rows(), j + full.cols(), A22(i, j));

        // for (int i : fixed_entry_)
        //     entries.emplace_back(i + full.rows(), i + full.cols(), 1);
        tmp = macro_full_to_reduced_grad(Eigen::MatrixXd::Ones(dim*dim, 1));
        for (int i = 0; i < tmp.size(); i++)
            entries.emplace_back(i + full.rows(), i + full.cols(), 1 - tmp(i));

        THessian mid(full.rows() + dof2, full.cols() + dof2);
        mid.setFromTriplets(entries.begin(), entries.end());

        NLProblem::full_hessian_to_reduced_hessian(mid, reduced);
    }

    NLHomoProblem::TVector NLHomoProblem::full_to_reduced(const TVector &full, const Eigen::MatrixXd &disp_grad) const
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();

        TVector reduced;
        reduced.setZero(dof1 + dof2);

        reduced.head(dof1) = NLProblem::full_to_reduced(full - io::Evaluator::generate_linear_field(state_.n_bases, state_.mesh_nodes, disp_grad));
        reduced.tail(dof2) = macro_full_to_reduced(utils::flatten(disp_grad));

        return reduced;
    }
    NLHomoProblem::TVector NLHomoProblem::full_to_reduced_grad(const TVector &full) const
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();

        TVector reduced;
        reduced.setZero(dof1 + dof2);

        reduced.head(dof1) = NLProblem::full_to_reduced_grad(full);
        reduced.tail(dof2) = macro_full_to_reduced_grad(constraint_grad) * full;

        return reduced;
    }
    NLHomoProblem::TVector NLHomoProblem::reduced_to_full(const TVector &reduced) const
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();

        Eigen::MatrixXd disp_grad = utils::unflatten(macro_reduced_to_full(reduced.tail(dof2)), dim);
        return NLProblem::reduced_to_full(reduced.head(dof1)) + io::Evaluator::generate_linear_field(state_.n_bases, state_.mesh_nodes, disp_grad);
    }
    Eigen::MatrixXd NLHomoProblem::reduced_to_disp_grad(const TVector &reduced) const
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();

        return utils::unflatten(macro_reduced_to_full(reduced.tail(dof2)), dim);
    }

    int NLHomoProblem::macro_reduced_size() const
    {
        const int dim = state_.mesh->dimension();
        if (only_symmetric)
            return ((dim + 1) * dim) / 2;
        else
            return dim * dim;
    }
    NLHomoProblem::TVector NLHomoProblem::macro_full_to_reduced(const TVector &full) const
    {
        if (only_symmetric)
            return full_to_symmetric.transpose() * full;
        else
            return full;
    }
    Eigen::MatrixXd NLHomoProblem::macro_full_to_reduced_grad(const Eigen::MatrixXd &full) const
    {
        Eigen::MatrixXd mid = full;
        for (int i : fixed_entry_)
            mid.row(i).setZero();

        if (only_symmetric)
            return symmetric_to_full.transpose() * mid;
        else
            return mid;
    }
    NLHomoProblem::TVector NLHomoProblem::macro_reduced_to_full(const TVector &reduced) const
    {
        if (only_symmetric)
            return symmetric_to_full * reduced;
        else
            return reduced;
    }
}
