#include "Parametrization.hpp"
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
    Eigen::VectorXd Parametrization::inverse_eval(const Eigen::VectorXd &y)
    {
        log_and_throw_error("Not supported");
        return Eigen::VectorXd();
    }
    
    Eigen::VectorXi IndexedParametrization::get_output_indexing(const Eigen::VectorXd &x) const
    {
        const int out_size = size(x.size());
        if (output_indexing_.size() == out_size)
            return output_indexing_;
        else if (output_indexing_.size() == 0)
        {
            Eigen::VectorXi ind;
            ind.setLinSpaced(out_size, 0, out_size - 1);
            return ind;
        }
        else
            log_and_throw_error(fmt::format("Indexing size and output size of the Parametrization do not match! {} vs {}", output_indexing_.size(), out_size));
        return Eigen::VectorXi();
    }
    
    int CompositeParametrization::size(const int x_size) const
    {
        int cur_size = x_size;
        for (const auto &p : parametrizations_)
            cur_size = p->size(cur_size);

        return cur_size;
    }

    Eigen::VectorXd CompositeParametrization::inverse_eval(const Eigen::VectorXd &y)
    {
        if (parametrizations_.empty())
            return y;

        Eigen::VectorXd x = y;
        for (int i = parametrizations_.size() - 1; i >= 0; i--)
        {
            x = parametrizations_[i]->inverse_eval(x);
        }

        return x;
    }

    Eigen::VectorXd CompositeParametrization::eval(const Eigen::VectorXd &x) const
    {
        if (parametrizations_.empty())
            return x;

        Eigen::VectorXd y = x;
        for (const auto &p : parametrizations_)
        {
            y = p->eval(y);
        }

        return y;
    }
    Eigen::VectorXd CompositeParametrization::apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
    {
        Eigen::VectorXd gradv = grad_full(get_output_indexing(x));

        if (parametrizations_.empty())
            return gradv;

        std::vector<Eigen::VectorXd> ys;
        auto y = x;
        for (const auto &p : parametrizations_)
        {
            ys.emplace_back(y);
            y = p->eval(y);
        }

        for (int i = parametrizations_.size() - 1; i >= 0; --i)
            gradv = parametrizations_[i]->apply_jacobian(gradv, ys[i]);

        return gradv;
    }
}