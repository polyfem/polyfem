#pragma once

#include <polyfem/Common.hpp>

#include <ipc/ipc.hpp>

namespace polyfem::solver
{
    class DiffCache
    {
    public:
        void init(const int ndof, const int n_time_steps = 0)
        {
            cur_size_ = 0;
            n_time_steps_ = n_time_steps;
            
            u_.setZero(ndof, n_time_steps + 1);
            if (n_time_steps_ > 0)
            {
                bdf_order_.setZero(n_time_steps + 1);
                v_.setZero(ndof, n_time_steps + 1);
                acc_.setZero(ndof, n_time_steps + 1);
                // gradu_h_prev_.resize(n_time_steps + 1);
            }
            gradu_h_.resize(n_time_steps + 1);

            contact_set_.resize(n_time_steps + 1);
            friction_constraint_set_.resize(n_time_steps + 1);
        }

        void cache_quantities_static(
            const Eigen::MatrixXd &u,
            const StiffnessMatrix &gradu_h,
            const ipc::CollisionConstraints &contact_set,
            const ipc::FrictionConstraints &friction_constraint_set)
        {
            u_ = u;

            gradu_h_[0] = gradu_h;
            contact_set_[0] = contact_set;
            friction_constraint_set_[0] = friction_constraint_set;

            cur_size_ = 1;
        }

        void cache_quantities_transient(
            const int cur_step,
            const int cur_bdf_order,
            const Eigen::MatrixXd &u,
            const Eigen::MatrixXd &v,
            const Eigen::MatrixXd &acc,
            const StiffnessMatrix &gradu_h,
            // const StiffnessMatrix &gradu_h_prev,
            const ipc::CollisionConstraints &contact_set,
            const ipc::FrictionConstraints &friction_constraint_set)
        {
            bdf_order_(cur_step) = cur_bdf_order;

            u_.col(cur_step) = u;
            v_.col(cur_step) = v;
            acc_.col(cur_step) = acc;

            gradu_h_[cur_step] = gradu_h;
            // gradu_h_prev_[cur_step] = gradu_h_prev;

            contact_set_[cur_step] = contact_set;
            friction_constraint_set_[cur_step] = friction_constraint_set;

            cur_size_++;
        }

        void cache_adjoints(const Eigen::MatrixXd &adjoint_mat) { adjoint_mat_ = adjoint_mat; }
        const Eigen::MatrixXd &adjoint_mat() const { return adjoint_mat_; }

        inline int size() const { return cur_size_; }
        inline int bdf_order(const int step) const { assert(step < size()); return bdf_order_(step); }

        Eigen::VectorXd u(const int step) const { assert(step < size()); return u_.col(step); }
        Eigen::VectorXd v(const int step) const { assert(step < size()); return v_.col(step); }
        Eigen::VectorXd acc(const int step) const { assert(step < size()); return acc_.col(step); }

        void cache_disp_grad(const Eigen::MatrixXd &disp_grad) { disp_grad_ = disp_grad; }
        Eigen::MatrixXd disp_grad() const { assert(disp_grad_.size() > 0); return disp_grad_; }

        const StiffnessMatrix &gradu_h(const int step) const { assert(step < size()); return gradu_h_[step]; }
        // const StiffnessMatrix &gradu_h_prev(const int step) const { assert(step < size()); return gradu_h_prev_[step]; }

        const ipc::CollisionConstraints &contact_set(const int step) const { assert(step < size()); return contact_set_[step]; }
        const ipc::FrictionConstraints &friction_constraint_set(const int step) const { assert(step < size()); return friction_constraint_set_[step]; }

    private:

        int n_time_steps_ = 0;
        int cur_size_ = 0;
    
        Eigen::MatrixXd u_; // PDE solution
        Eigen::MatrixXd v_; // velocity in transient elastic simulations
        Eigen::MatrixXd acc_; // acceleration in transient elastic simulations

        Eigen::MatrixXd disp_grad_; // macro linear displacement in homogenization

        Eigen::VectorXi bdf_order_; // BDF orders used at each time step in forward simulation
        
        std::vector<StiffnessMatrix> gradu_h_; // gradient of force at time T wrt. u  at time T
        // std::vector<StiffnessMatrix> gradu_h_prev_; // gradient of force at time T wrt. u at time (T-1) in transient simulations

        std::vector<ipc::CollisionConstraints> contact_set_;
        std::vector<ipc::FrictionConstraints> friction_constraint_set_;

        Eigen::MatrixXd adjoint_mat_;
    };
}