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
                const double t, const std::vector<std::shared_ptr<Form>> &forms, 
				  const std::shared_ptr<PeriodicContactForm> &contact_form) : NLProblem(full_size, boundary_nodes, local_boundary, n_boundary_samples, rhs_assembler, state, t, forms), contact_form_(contact_form)
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
    
    Eigen::VectorXd NLHomoProblem::reduced_to_extended(const Eigen::VectorXd &reduced) const
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();
        assert(reduced.size() == dof1 + dof2);
        
        Eigen::VectorXd fluctuation = NLProblem::reduced_to_full(reduced.head(dof1));
        Eigen::VectorXd disp_grad = macro_reduced_to_full(reduced.tail(dof2));
        Eigen::VectorXd extended(fluctuation.size() + disp_grad.size());
        extended << fluctuation, disp_grad;

        return extended;
    }
    
    Eigen::VectorXd NLHomoProblem::extended_to_reduced_grad(const Eigen::VectorXd &extended) const
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();

        Eigen::VectorXd grad(dof1 + dof2);
        grad.head(dof1) = NLProblem::full_to_reduced_grad(extended.head(extended.size() - dim * dim));
        grad.tail(dof2) = macro_full_to_reduced_grad(extended.tail(dim * dim));

        return grad;
    }

    double NLHomoProblem::value(const TVector &x)
    {
        double val = NLProblem::value(x);
        if (contact_form_)
            val += contact_form_->value(reduced_to_extended(x));

        return val;
    }
    void NLHomoProblem::gradient(const TVector &x, TVector &gradv)
    {
        NLProblem::gradient(x, gradv);

        if (contact_form_)
        {
            Eigen::VectorXd grad_extended;
            contact_form_->first_derivative(reduced_to_extended(x), grad_extended);
            gradv += extended_to_reduced_grad(grad_extended);
        }
    }
    void NLHomoProblem::extended_hessian_to_reduced_hessian(const THessian &extended, THessian &reduced) const
    {
        const int dim = state_.mesh->dimension();
        const int dof2 = macro_reduced_size();
        const int dof1 = reduced_size();

        Eigen::MatrixXd A12, A22;
        {
            Eigen::MatrixXd tmp = Eigen::MatrixXd(extended.rightCols(dim * dim));
            A12 = macro_full_to_reduced_grad(tmp.topRows(tmp.rows() - dim * dim).transpose()).transpose();
            A22 = macro_full_to_reduced_grad(macro_full_to_reduced_grad(tmp.bottomRows(dim * dim)).transpose());
        }

        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(extended.nonZeros() + A12.size() * 2 + A22.size());

        for (int k = 0; k < full_size_; ++k)
            for (StiffnessMatrix::InnerIterator it(extended, k); it; ++it)
                if (it.row() < full_size_ && it.col() < full_size_)
                    entries.emplace_back(it.row(), it.col(), it.value());

        for (int i = 0; i < A12.rows(); i++)
        {
            for (int j = 0; j < A12.cols(); j++)
            {
                entries.emplace_back(i, full_size_ + j, A12(i, j));
                entries.emplace_back(full_size_ + j, i, A12(i, j));
            }
        }
        
        for (int i = 0; i < A22.rows(); i++)
            for (int j = 0; j < A22.cols(); j++)
                entries.emplace_back(i + full_size_, j + full_size_, A22(i, j));

        Eigen::VectorXd tmp = macro_full_to_reduced_grad(Eigen::VectorXd::Ones(dim*dim));
        for (int i = 0; i < tmp.size(); i++)
            entries.emplace_back(i + full_size_, i + full_size_, 1 - tmp(i));

        THessian mid(full_size_ + dof2, full_size_ + dof2);
        mid.setFromTriplets(entries.begin(), entries.end());

        NLProblem::full_hessian_to_reduced_hessian(mid, reduced);
    }
    void NLHomoProblem::hessian(const TVector &x, THessian &hessian)
    {
        NLProblem::hessian(x, hessian);

        if (contact_form_)
        {
            THessian hess_extended;
            contact_form_->second_derivative(reduced_to_extended(x), hess_extended);

            THessian hess;
            extended_hessian_to_reduced_hessian(hess_extended, hess);
            hessian += hess;
        }
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

    bool NLHomoProblem::is_step_valid(const TVector &x0, const TVector &x1) const
    {
        if (contact_form_)
            return NLProblem::is_step_valid(x0, x1) && contact_form_->is_step_valid(reduced_to_extended(x0), reduced_to_extended(x1));
        else
            return NLProblem::is_step_valid(x0, x1);
    }
    bool NLHomoProblem::is_step_collision_free(const TVector &x0, const TVector &x1) const
    {
        if (contact_form_)
            return NLProblem::is_step_collision_free(x0, x1) && contact_form_->is_step_collision_free(reduced_to_extended(x0), reduced_to_extended(x1));
        else
            return NLProblem::is_step_collision_free(x0, x1);
    }
    double NLHomoProblem::max_step_size(const TVector &x0, const TVector &x1) const
    {
        if (contact_form_)
            return std::min(NLProblem::max_step_size(x0, x1), contact_form_->max_step_size(reduced_to_extended(x0), reduced_to_extended(x1)));
        else
            return NLProblem::max_step_size(x0, x1);
    }

    void NLHomoProblem::line_search_begin(const TVector &x0, const TVector &x1)
    {
        NLProblem::line_search_begin(x0, x1);
        if (contact_form_)
            contact_form_->line_search_begin(reduced_to_extended(x0), reduced_to_extended(x1));
    }
    void NLHomoProblem::post_step(const int iter_num, const TVector &x)
    {
        NLProblem::post_step(iter_num, x);
        if (contact_form_)
            contact_form_->post_step(iter_num, reduced_to_extended(x));
    }

    void NLHomoProblem::solution_changed(const TVector &new_x)
    {
        NLProblem::solution_changed(new_x);
        if (contact_form_)
            contact_form_->solution_changed(reduced_to_extended(new_x));
    }

    void NLHomoProblem::init_lagging(const TVector &x)
    {
        NLProblem::init_lagging(x);
        if (contact_form_)
            contact_form_->init_lagging(reduced_to_extended(x));
    }
    void NLHomoProblem::update_lagging(const TVector &x, const int iter_num)
    {
        NLProblem::update_lagging(x, iter_num);
        if (contact_form_)
            contact_form_->update_lagging(reduced_to_extended(x), iter_num);
    }

    void NLHomoProblem::update_quantities(const double t, const TVector &x)
    {
        NLProblem::update_quantities(t, x);
        if (contact_form_)
            contact_form_->update_quantities(t, reduced_to_extended(x));
    }
}
