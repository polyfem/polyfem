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
            if (material_params["volume_selection"].size() > 0)
            {
                for (int i : material_params["volume_selection"])
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

            apply_log = type.find("log") != std::string::pos;
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
            Eigen::VectorXd x = reduced;

            if (use_E_nu)
                x = E_nu_to_lambda_mu(x);
            
            if (apply_log)
                x = x.array().exp().eval();

            if (per_body)
                x = per_body_to_per_elem(x);
            
            state->assembler.update_lame_params(x.segment(0, n_elem), cur_mus);
        }

    private:
        const State &state_;
        std::map<int, std::array<int, 2>> body_id_map; // from body_id to {elem_id, index}
        bool apply_log, use_E_nu, per_body;
        int n_elem;

        Eigen::VectorXd per_body_to_per_elem(const Eigen::VectorXd &x)
        {
            assert(x.size() == reduced_size_);
            if (per_body)
            {
                Eigen::VectorXd y(full_size_);
                
                for (int e = 0; e < n_elem; e++)
                {
                    const int body_id = state.mesh->get_body_id(e);
                    y(e) = x(body_id_map.at(body_id)[1]); // lambda or E
                    y(e + n_elem) = x(body_id_map.at(body_id)[1] + body_id_map.size()); // mu or nu
                }

                return y;
            }
            
            return x;
        }

        Eigen::VectorXd E_nu_to_lambda_mu(const Eigen::VectorXd &x)
        {
            const int size = x.size() / 2;
            assert(size * 2 == x.size());

            Eigen::VectorXd y(x.size());
            for (int i = 0; i < size; i++)
            {
                y(i) = convert_to_lambda(state.mesh->is_volume(), x(i), x(i + size));
                y(i + size) = convert_to_mu(x(i), x(i + size));
            }
            
            return y;
        }
    }
}