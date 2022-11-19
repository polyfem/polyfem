#pragma once

#include "Constraints.hpp"
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

namespace polyfem
{
	class MaterialConstraints : public Constraints
	{
	public:
        MaterialConstraints(const json &constraint_params, const State &state): Constraints(constraint_params, 0, 0), state_(state)
        {
            std::string type = constraint_params["restriction"];

            std::set<int> optimize_body_ids;
            if (constraint_params["volume_selection"].size() > 0)
            {
                for (int i : constraint_params["volume_selection"])
                    optimize_body_ids.insert(i);
            }
            else
                logger().info("No optimization body specified, optimize material of every mesh...");
        
            const int dim = state_.mesh->dimension();
            n_elem = state_.bases.size();

            int n = 0;
            for (int e = 0; e < n_elem; e++)
            {
                const int body_id = state_.mesh->get_body_id(e);
                const bool found_body_id = optimize_body_ids.count(body_id) || optimize_body_ids.size() == 0;
                if (!body_id_map.count(body_id) && found_body_id)
                {
                    body_id_map[body_id] = {{e, n}};
                    n++;
                }
            }

            apply_log = type.find("log") != std::string::npos;
            use_E_nu = type.find("E_nu") != std::string::npos;

            full_size_ = n_elem * 2;
            if (type.find("constant") != std::string::npos)
            {
                reduced_size_ = body_id_map.size() * 2;
                per_body = true;
            }
            else
            {
                reduced_size_ = full_size_;
                per_body = false;
            }
        }

        void update_state(std::shared_ptr<State> state, const Eigen::VectorXd &reduced) override
        {
            Eigen::VectorXd x = reduced_to_full(reduced);
            
            state->assembler.update_lame_params(x.segment(0, n_elem), x.segment(n_elem, n_elem));
        }

		Eigen::VectorXd reduced_to_full(const Eigen::VectorXd &reduced) const
        {
            Eigen::VectorXd x = reduced;

            if (apply_log)
            {
                if (use_E_nu)
                    x.segment(0, x.size()/2) = x.segment(0, x.size()/2).array().exp().eval();
                else
                    x = x.array().exp().eval();
            }

            if (use_E_nu)
                x = E_nu_to_lambda_mu(x);

            if (per_body)
                x = per_body_to_per_elem(x);
            
            return x;
        }
		
        Eigen::VectorXd full_to_reduced(const Eigen::VectorXd &full) const
        {
            Eigen::VectorXd x = full;

            if (per_body)
                x = per_elem_to_per_body(x);
            
            if (use_E_nu)
                x = lambda_mu_to_E_nu(x);

            if (apply_log)
            {
                if (use_E_nu)
                    x.segment(0, x.size()/2) = x.segment(0, x.size()/2).array().log().eval();
                else
                    x = x.array().log().eval();
            }

            return x;
        }

        Eigen::VectorXd x_from_state() const
        {
            Eigen::VectorXd x(full_size_);
            x.segment(0, n_elem) = state_.assembler.lame_params().lambda_mat_;
            x.segment(n_elem, n_elem) = state_.assembler.lame_params().mu_mat_;

            return full_to_reduced(x);
        }

        Eigen::VectorXd grad_full_to_grad_reduced(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &reduced) const
        {
            Eigen::VectorXd grad = grad_full;

            if (per_body)
                grad = per_body_to_per_elem_grad(grad);

            if (use_E_nu)
            {
                Eigen::VectorXd E_nu = reduced;
                if (apply_log)
                    E_nu.segment(0, E_nu.size() / 2) = E_nu.segment(0, E_nu.size() / 2).array().exp().eval();
                grad = E_nu_to_lambda_mu_grad(grad, E_nu);
            }

            if (apply_log)
            {
                if (use_E_nu)
                    grad.segment(0, grad.size()/2) = (reduced.segment(0, grad.size()/2).array().exp() * grad.segment(0, grad.size()/2).array()).eval();
                else
                    grad = (reduced.array().exp() * grad.array()).eval();
            }

            return grad;
        }

    private:
        const State &state_;
        std::map<int, std::array<int, 2>> body_id_map; // from body_id to {elem_id, index}
        bool apply_log, use_E_nu, per_body;
        int n_elem;

        Eigen::VectorXd per_body_to_per_elem(const Eigen::VectorXd &x) const
        {
            assert(x.size() == reduced_size_);
            Eigen::VectorXd y(full_size_);
            
            for (int e = 0; e < n_elem; e++)
            {
                const int body_id = state_.mesh->get_body_id(e);
                y(e) = x(body_id_map.at(body_id)[1]); // lambda or E
                y(e + n_elem) = x(body_id_map.at(body_id)[1] + body_id_map.size()); // mu or nu
            }

            return y;
        }

        Eigen::VectorXd per_body_to_per_elem_grad(const Eigen::VectorXd &grad_elem) const
        {
            assert(grad_elem.size() == full_size_);
            Eigen::VectorXd grad_body;
            grad_body.setZero(reduced_size_);

            for (int e = 0; e < n_elem; e++)
            {
                const int body_id = state_.mesh->get_body_id(e);
                grad_body(body_id) += grad_elem(e);
                grad_body(body_id + body_id_map.size()) += grad_elem(e + n_elem);
            }

            return grad_body;
        }

        Eigen::VectorXd per_elem_to_per_body(const Eigen::VectorXd &x) const
        {
            assert(x.size() == full_size_);
            Eigen::VectorXd y(reduced_size_);
            
            for (auto i : body_id_map)
            {
                y(i.second[1]) = x(i.second[0]);
                y(i.second[1] + body_id_map.size()) = x(i.second[0] + n_elem);
            }

            return y;
        }

        Eigen::VectorXd E_nu_to_lambda_mu(const Eigen::VectorXd &x) const
        {
            const int size = x.size() / 2;
            assert(size * 2 == x.size());

            Eigen::VectorXd y(x.size());
            for (int i = 0; i < size; i++)
            {
                y(i) = convert_to_lambda(state_.mesh->is_volume(), x(i), x(i + size));
                y(i + size) = convert_to_mu(x(i), x(i + size));
            }
            
            return y;
        }

        Eigen::VectorXd E_nu_to_lambda_mu_grad(const Eigen::VectorXd &grad_lambda_mu, const Eigen::VectorXd &E_nu) const
        {
            const int size = grad_lambda_mu.size() / 2;
            assert(size * 2 == grad_lambda_mu.size());
            assert(size * 2 == E_nu.size());

            Eigen::VectorXd grad_E_nu;
            grad_E_nu.setZero(grad_lambda_mu.size());
            for (int i = 0; i < size; i++)
            {
                Eigen::Matrix2d jacobian = d_lambda_mu_d_E_nu(state_.mesh->is_volume(), E_nu(i), E_nu(i + size));
                grad_E_nu(i) = grad_lambda_mu(i) * jacobian(0, 0) + grad_lambda_mu(i + size) * jacobian(1, 0);
                grad_E_nu(i + size) = grad_lambda_mu(i) * jacobian(0, 1) + grad_lambda_mu(i + size) * jacobian(1, 1);
            }
            
            return grad_E_nu;
        }
    
        Eigen::VectorXd lambda_mu_to_E_nu(const Eigen::VectorXd &x) const
        {
            const int size = x.size() / 2;
            assert(size * 2 == x.size());

            Eigen::VectorXd y(x.size());
            for (int i = 0; i < size; i++)
            {
                y(i) = convert_to_E(state_.mesh->is_volume(), x(i), x(i + size));
                y(i + size) = convert_to_nu(state_.mesh->is_volume(), x(i), x(i + size));
            }
            
            return y;
        }
    };
}