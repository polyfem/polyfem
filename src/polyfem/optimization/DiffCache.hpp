#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <ipc/ipc.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/collisions/tangential/tangential_collisions.hpp>

namespace polyfem::solver
{
	enum class CacheLevel
	{
		None,
		Solution,
		Derivatives
	};

	class DiffCache
	{
	public:
		void init(const int dimension, const int ndof, const int n_time_steps = 0)
		{
			cur_size_ = 0;
			n_time_steps_ = n_time_steps;

			u_.setZero(ndof, n_time_steps + 1);
			disp_grad_.assign(n_time_steps + 1, Eigen::MatrixXd::Zero(dimension, dimension));
			if (n_time_steps_ > 0)
			{
				bdf_order_.setZero(n_time_steps + 1);
				v_.setZero(ndof, n_time_steps + 1);
				acc_.setZero(ndof, n_time_steps + 1);
				// gradu_h_prev_.resize(n_time_steps + 1);
			}
			gradu_h_.resize(n_time_steps + 1);
			collision_set_.resize(n_time_steps + 1);
			smooth_collision_set_.resize(n_time_steps + 1);
			friction_collision_set_.resize(n_time_steps + 1);
			normal_adhesion_collision_set_.resize(n_time_steps + 1);
			tangential_adhesion_collision_set_.resize(n_time_steps + 1);
		}

		void cache_quantities_static(
			const Eigen::MatrixXd &u,
			const StiffnessMatrix &gradu_h,
			const ipc::NormalCollisions &collision_set,
			const ipc::SmoothCollisions &smooth_collision_set,
			const ipc::TangentialCollisions &friction_constraint_set,
			const ipc::NormalCollisions &normal_adhesion_set,
			const ipc::TangentialCollisions &tangential_adhesion_set,
			const Eigen::MatrixXd &disp_grad)
		{
			u_ = u;

			gradu_h_[0] = gradu_h;
			collision_set_[0] = collision_set;
			smooth_collision_set_[0] = smooth_collision_set;
			friction_collision_set_[0] = friction_constraint_set;
			normal_adhesion_collision_set_[0] = normal_adhesion_set;
			tangential_adhesion_collision_set_[0] = tangential_adhesion_set;
			disp_grad_[0] = disp_grad;

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
			const ipc::NormalCollisions &collision_set,
			const ipc::SmoothCollisions &smooth_collision_set,
			const ipc::TangentialCollisions &friction_collision_set)
		{
			bdf_order_(cur_step) = cur_bdf_order;

			u_.col(cur_step) = u;
			v_.col(cur_step) = v;
			acc_.col(cur_step) = acc;

			gradu_h_[cur_step] = gradu_h;
			// gradu_h_prev_[cur_step] = gradu_h_prev;

			collision_set_[cur_step] = collision_set;
			smooth_collision_set_[cur_step] = smooth_collision_set;
			friction_collision_set_[cur_step] = friction_collision_set;

			cur_size_++;
		}

		void cache_quantities_quasistatic(
			const int cur_step,
			const Eigen::MatrixXd &u,
			const StiffnessMatrix &gradu_h,
			const ipc::NormalCollisions &collision_set,
			const ipc::SmoothCollisions &smooth_collision_set,
			const ipc::NormalCollisions &normal_adhesion_set,
			const Eigen::MatrixXd &disp_grad)
		{
			u_.col(cur_step) = u;
			gradu_h_[cur_step] = gradu_h;
			collision_set_[cur_step] = collision_set;
			smooth_collision_set_[cur_step] = smooth_collision_set;
			normal_adhesion_collision_set_[cur_step] = normal_adhesion_set;
			disp_grad_[cur_step] = disp_grad;

			cur_size_++;
		}

		void cache_adjoints(const Eigen::MatrixXd &adjoint_mat) { adjoint_mat_ = adjoint_mat; }
		const Eigen::MatrixXd &adjoint_mat() const { return adjoint_mat_; }

		inline int size() const { return cur_size_; }
		inline int bdf_order(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += bdf_order_.size();
			return bdf_order_(step);
		}

		Eigen::MatrixXd disp_grad(int step = 0) const
		{
			assert(step < size());
			if (step < 0)
				step += disp_grad_.size();
			return disp_grad_[step];
		}

		Eigen::VectorXd u(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += u_.cols();
			return u_.col(step);
		}
		Eigen::VectorXd v(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += v_.cols();
			return v_.col(step);
		}
		Eigen::VectorXd acc(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += acc_.cols();
			return acc_.col(step);
		}

		const StiffnessMatrix &gradu_h(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += gradu_h_.size();
			return gradu_h_[step];
		}
		// const StiffnessMatrix &gradu_h_prev(const int step) const { assert(step < size()); return gradu_h_prev_[step]; }

		const ipc::NormalCollisions &collision_set(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += collision_set_.size();
			return collision_set_[step];
		}
		const ipc::SmoothCollisions &smooth_collision_set(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += smooth_collision_set_.size();
			return smooth_collision_set_[step];
		}
		const ipc::TangentialCollisions &friction_collision_set(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += friction_collision_set_.size();
			return friction_collision_set_[step];
		}
		const ipc::NormalCollisions &normal_adhesion_collision_set(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += normal_adhesion_collision_set_.size();
			return normal_adhesion_collision_set_[step];
		}
		const ipc::TangentialCollisions &tangential_adhesion_collision_set(int step) const
		{
			assert(step < size());
			if (step < 0)
				step += tangential_adhesion_collision_set_.size();
			return tangential_adhesion_collision_set_[step];
		}

	private:
		int n_time_steps_ = 0;
		int cur_size_ = 0;

		std::vector<Eigen::MatrixXd> disp_grad_; // macro linear displacement in homogenization
		Eigen::MatrixXd u_;                      // PDE solution
		Eigen::MatrixXd v_;                      // velocity in transient elastic simulations
		Eigen::MatrixXd acc_;                    // acceleration in transient elastic simulations

		Eigen::VectorXi bdf_order_; // BDF orders used at each time step in forward simulation

		std::vector<StiffnessMatrix> gradu_h_; // gradient of force at time T wrt. u  at time T
		// std::vector<StiffnessMatrix> gradu_h_prev_; // gradient of force at time T wrt. u at time (T-1) in transient simulations

		std::vector<ipc::NormalCollisions> collision_set_;
		std::vector<ipc::SmoothCollisions> smooth_collision_set_;
		std::vector<ipc::TangentialCollisions> friction_collision_set_;

		std::vector<ipc::NormalCollisions> normal_adhesion_collision_set_;
		std::vector<ipc::TangentialCollisions> tangential_adhesion_collision_set_;

		Eigen::MatrixXd adjoint_mat_;
	};
} // namespace polyfem::solver