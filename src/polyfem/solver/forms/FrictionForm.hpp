#pragma once

#include "Form.hpp"

#include <polyfem/State.hpp>

#include <polyfem/utils/Types.hpp>

#include <ipc/collision_mesh.hpp>
#include <ipc/friction/friction_constraint.hpp>

namespace polyfem
{
	namespace solver
	{
		class FrictionForm : public Form
		{
		public:
			FrictionForm(const State &state, const double epsv, const double mu, const double dt, const ipc::CollisionMesh &collision_mesh);

			double value(const Eigen::VectorXd &x) override;
			void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
			void hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

			void init_lagging(const Eigen::VectorXd &x) override;
			void update_lagging(const Eigen::VectorXd &x) override;

		private:
			const double epsv_;
			const double mu_;
			const double dt_;
			const ipc::CollisionMesh &collision_mesh_;
			ipc::FrictionConstraints friction_constraint_set_;
			Eigen::MatrixXd displaced_prev_; ///< @brief Displaced vertices at the start of the time-step.

			const State &state_;
		};
	} // namespace solver
} // namespace polyfem
