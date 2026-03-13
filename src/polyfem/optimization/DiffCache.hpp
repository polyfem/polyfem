#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/State.hpp>

#include <polyfem/utils/Types.hpp>
#include <polyfem/optimization/CacheLevel.hpp>

#include <ipc/ipc.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/collisions/tangential/tangential_collisions.hpp>

#include <Eigen/Core>

#include <cassert>
#include <vector>

namespace polyfem
{
	/// @brief Storage for additional data required by differntial code.
	class DiffCache
	{
	public:
		void cache_adjoints(const Eigen::MatrixXd &adjoint_mat);

		/// @brief Cache time-dependent adjoint optimization data.
		/// @param[in] step Current time step.
		/// @param[in] state Current forward simulation state. Will mutate it to get latest value.
		/// @param[in] sol Current solution.
		/// @param[in] disp_grad Pointer to displacement gradient matrix. Assumes zero if nullptr.
		/// @param[in] pressure Pointer to pressure matrix. PASS nullptr ONLY.
		///
		/// @warning We DO NOT support navier stoke problem yet!! Passing non-null pressure triggers exception.
		void cache_transient(
			int step,
			State &state,
			const Eigen::MatrixXd &sol,
			const Eigen::MatrixXd *disp_grad,
			const Eigen::MatrixXd *pressure);

		const Eigen::MatrixXd &adjoint_mat() const { return adjoint_mat_; }

		const StiffnessMatrix &basis_nodes_to_gbasis_nodes() const;

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

		// Mapping from positions of FE basis nodes to positions of geometry nodes.
		StiffnessMatrix basis_nodes_to_gbasis_nodes_;

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

		void init(const int dimension, const int ndof, const int n_time_steps = 0);

		void cache_quantities_static(
			const Eigen::MatrixXd &u,
			const StiffnessMatrix &gradu_h,
			const ipc::NormalCollisions &collision_set,
			const ipc::SmoothCollisions &smooth_collision_set,
			const ipc::TangentialCollisions &friction_constraint_set,
			const ipc::NormalCollisions &normal_adhesion_set,
			const ipc::TangentialCollisions &tangential_adhesion_set,
			const Eigen::MatrixXd &disp_grad);

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
			const ipc::TangentialCollisions &friction_collision_set);

		void cache_quantities_quasistatic(
			const int cur_step,
			const Eigen::MatrixXd &u,
			const StiffnessMatrix &gradu_h,
			const ipc::NormalCollisions &collision_set,
			const ipc::SmoothCollisions &smooth_collision_set,
			const ipc::NormalCollisions &normal_adhesion_set,
			const Eigen::MatrixXd &disp_grad);
	};
} // namespace polyfem
